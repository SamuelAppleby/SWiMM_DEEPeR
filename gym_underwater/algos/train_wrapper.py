import csv
import os
import time
from collections import deque
import numpy as np


class AlgWrapper:
    def __init__(self):
        self.best_n_episodes = 100
        self.episode_rewards = [0.0]
        self.episode_reward = np.zeros((1,))
        self.ep_info_buf = deque(maxlen=100)
        self.n_updates = 0
        self.infos_values = []
        self.mb_infos_vals = []
        self.mean_reward = -np.inf
        self.best_mean_reward = -np.inf
        self.ep_len = 0

    def create_callback(self, algo, save_path, verbose=1):
        """
        Create callback function for saving best model frequently and stopping run on reward threshold.

        :param algo: (str)
        :param save_path: (str)
        :param verbose: (int)
        :return: (function) the callback function
        """
        if algo != 'sac':
            raise NotImplementedError("Callback creation not implemented yet for {}".format(algo))

        def sac_callback(_locals, _globals):
            """
            Callback for saving best model when using SAC. Early stopping also implemented here.

            :param _locals: (dict)
            :param _globals: (dict)
            :return: (bool) If False: stop training
            """
            frac = 1.0 - _locals['num_collected_steps'] / _locals['total_timesteps']
            self.ep_len += 1
            self.episode_rewards[-1] += _locals['rewards'][-1]

            if _locals['log_interval'] > 0 and self.ep_len % _locals['log_interval'] == 0 and self.ep_len > 0:
                print('{} steps'.format(self.ep_len))

            if _locals['dones'][-1]:
                print("Episode finished. \nReward: {:.2f} \nSteps: {}".format(self.episode_rewards[-1], self.ep_len))

                _locals['self'].logger.record("train/episode_reward", self.episode_rewards[-1])
                _locals['self'].logger.record("train/episode_length", self.ep_len)
                # _locals['self'].logger.record("train/episode_termination", int(ep_len == _locals['train_freq']))
                # _locals['self'].logger.record("train/training_reward", np.sum(episode_rewards))

                mean_reward = round(float(np.mean(self.episode_rewards[-self.best_n_episodes:])), 1) if len(self.episode_rewards) > self.best_n_episodes else round(
                    float(np.mean(self.episode_rewards)), 1)

                if mean_reward > self.best_mean_reward:
                    best_mean_reward = mean_reward
                    begin_time = time.time()
                    print("Saving best model ...")
                    _locals['self'].save(os.path.join(_locals['self'].tensorboard_log, "bestmodel"))

                    with open(os.path.join(_locals['self'].tensorboard_log, "ep_nums_for_best.csv"), 'a') as csv_file:
                        best_writer = csv.writer(csv_file)
                        best_writer.writerow([len(self.episode_rewards)])
                        csv_file.close()

                    print("Model saved, time taken: ", time.time() - begin_time)

                # if _locals['self'].verbose >= 1 and _locals['log_interval'] is not None and len(episode_rewards) % _locals['log_interval'] == 0:
                #     logger.logkv("episodes", len(episode_rewards))
                #     logger.logkv("mean {} episode reward".format(best_n_episodes), mean_reward)
                #     logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                #     logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                #     logger.logkv("n_updates", self.n_updates)
                #     logger.logkv("current_lr", current_lr)
                #     logger.logkv("fps", fps)
                #     logger.logkv('time_elapsed', "{:.2f}".format(time.time() - start_time))
                #     if len(infos_values) > 0:
                #         for (name, val) in zip(self.infos_names, infos_values):
                #             logger.logkv(name, val)
                #     logger.logkv("total timesteps", step)
                #     logger.dumpkvs()
                #     # Reset infos:
                #     infos_values = []

                self.episode_rewards.append(0.0)
                ep_len = 0

            return True

        return sac_callback
