import csv
import os
import time
from collections import deque
import numpy as np

from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common import TensorboardWriter
import tensorflow as tf


class SACWrap(SAC):
    """
    User friendly high level version of algorithm
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None):
        super().__init__(policy, env, gamma, learning_rate, buffer_size, learning_starts, train_freq, batch_size, tau, ent_coef, target_update_interval, gradient_steps, target_entropy, action_noise,
                         random_exploration, verbose, tensorboard_log, _init_setup_model, policy_kwargs, full_tensorboard_log, seed, n_cpu_tf_sess)
        self.n_updates = None

    def optimize(self, step, writer, current_lr):
        """
        Do several optimization steps to update the different networks.
        :param step: (int) current timestep
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) Current learning rate
        :return: ([np.ndarray]) values used for monitoring
        """
        train_start = time.time()
        mb_infos_vals = []

        if step+1 >= self.batch_size and step+1 >= self.learning_starts:
            for grad_step in range(self.gradient_steps):
                self.n_updates += 1
                # Update policy and critics (q functions)
                mb_infos_vals.append(self._train_step(step, writer, current_lr))

                if (step + grad_step) % self.target_update_interval == 0:
                    # Update target network
                    self.sess.run(self.target_update_op)

        if self.n_updates > 0:
            print("SAC training duration: {:.2f}s".format(time.time() - train_start))
        return mb_infos_vals

    def learn(self, total_timesteps, callback=None,
              log_interval=1, tb_log_name="SAC", print_freq=100):
        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:

            best_n_episodes = 100
            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)

            start_time = time.time()
            episode_rewards = [0.0]

            obs = self.env.reset()

            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            self.n_updates = 0
            infos_values = []
            mb_infos_vals = []
            mean_reward = -np.inf
            best_mean_reward = -np.inf
            ep_len = 0

            for step in range(total_timesteps):

                # print('Step {} (total):'.format(step))
                # Compute current learning_rate
                frac = 1.0 - step / total_timesteps
                current_lr = self.learning_rate(frac, mean_reward)

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if step < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert rescaled_action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                ep_len += 1
                episode_rewards[-1] += reward

                if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
                    print('{} steps'.format(ep_len))

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, rescaled_action, reward, new_obs, float(done))
                obs = new_obs

                # Done check here as the last step may have been force reset, don't want to do it twice
                if not done and ep_len == self.train_freq:
                    print('Maximum episode length reached')
                    obs = self.env.reset()
                    done = True

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                if done:
                    print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))

                    if writer is not None:
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=episode_rewards[-1])]), step)
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_length", simple_value=ep_len)]), len(episode_rewards)-1)

                        val = 1 if ep_len == self.train_freq else 0
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_termination", simple_value=val)]), len(episode_rewards)-1)
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="training_reward", simple_value=np.sum(episode_rewards))]), step)

                    mb_infos_vals = self.optimize(step, writer, current_lr)

                    if len(episode_rewards) > best_n_episodes:
                        mean_reward = round(float(np.mean(episode_rewards[-best_n_episodes:])), 1)

                        if mean_reward > best_mean_reward:
                            begin_time = time.time()
                            print("Saving best model ...")
                            self.save(save_path=os.path.join(self.tensorboard_log, "bestmodel"), cloudpickle=True)

                            best_mean_reward = mean_reward
                            with open(os.path.join(self.tensorboard_log, "ep_nums_for_best.csv"), 'a') as csv_file:
                                best_writer = csv.writer(csv_file)
                                best_writer.writerow([len(episode_rewards)])
                                csv_file.close()
                            print("Model saved, time taken: ", time.time() - begin_time)

                    if self.verbose >= 1 and log_interval is not None and len(episode_rewards) % log_interval == 0:
                        fps = int(step / (time.time() - start_time))
                        logger.logkv("episodes", len(episode_rewards))
                        logger.logkv("mean {} episode reward".format(best_n_episodes), mean_reward)
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                        logger.logkv("n_updates", self.n_updates)
                        logger.logkv("current_lr", current_lr)
                        logger.logkv("fps", fps)
                        logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                        if len(infos_values) > 0:
                            for (name, val) in zip(self.infos_names, infos_values):
                                logger.logkv(name, val)
                        logger.logkv("total timesteps", step)
                        logger.dumpkvs()
                        # Reset infos:
                        infos_values = []

                    episode_rewards.append(0.0)
                    ep_len = 0

            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)
        return self
