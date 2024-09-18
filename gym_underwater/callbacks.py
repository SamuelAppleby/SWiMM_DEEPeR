import csv

from stable_baselines3.common import base_class
import os
from typing import Union, Optional, Tuple, Callable, Dict, Any, List
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import should_collect_more_steps
from tqdm import tqdm

import gymnasium

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.type_aliases import TrainFrequencyUnit, TrainFreq
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from .enums import EpisodeTerminationType
import psutil


def convert_train_freq(train_freq) -> TrainFreq:
    """
    Convert `train_freq` parameter (int or tuple)
    to a TrainFreq object.
    """
    if isinstance(train_freq, TrainFreq):
        return train_freq

    # The value of the train frequency will be checked later
    if not isinstance(train_freq, tuple):
        train_freq = (train_freq, 'step')
    try:
        train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
    except ValueError as e:
        raise ValueError(
            f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
        ) from e

    if not isinstance(train_freq[0], int):
        raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

    return TrainFreq(*train_freq)


# Code adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L337.
def should_collect_more_steps_vec(
        train_freq: TrainFreq,
        num_collected_steps: np.ndarray[int],
        num_collected_episodes: np.ndarray[int],
        count_targets: np.ndarray[int]
) -> bool:
    """
    Helper used in ``collect_rollouts()`` of off-policy algorithms
    to determine the termination condition.

    :param train_freq: How much experience should be collected before updating the policy.
    :param num_collected_steps: The number of already collected steps.
    :param num_collected_episodes: The number of already collected episodes.
    :param count_targets: The number of targets to collect.
    :return: Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return bool((num_collected_steps < count_targets).any())

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return bool((num_collected_episodes < count_targets).any())

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def validate_episode_termination(info):
    if info['episode_termination_type'] is None:
        assert info["TimeLimit.truncated"]
        info['episode_termination_type'] = EpisodeTerminationType.THRESHOLD_REACHED


class SwimCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.is_monitor_wrapped = False

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # At this point we are guaranteed to have training information (as train() has just been called)
        if self.model._n_updates > 0:
            self.logger.dump(self.num_timesteps)

        eval_callback = [obj for obj in self.locals['callback'].callbacks if isinstance(obj, SwimEvalCallback)]

        if len(eval_callback) > 0:
            eval_callback = eval_callback[0]

            # If an evaluation is about to be injected, don't resume physics
            if (self.num_timesteps > eval_callback.min_train_steps) and (eval_callback.n_rollout_calls > 0) and ((eval_callback.n_rollout_calls % eval_callback.eval_freq) == 0):
                return

        for env in self.training_env.envs:
            env.unwrapped.on_rollout_start()

        return

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        for i in range(self.training_env.num_envs):
            if self.training_env.buf_dones[i]:
                if self.is_monitor_wrapped:
                    print('[TRAINING] Episode finished. \nReward: {:.2f} \nSteps: {}'.format(self.training_env.envs[i].get_wrapper_attr('episode_returns')[-1],
                                                                                             self.training_env.envs[i].get_wrapper_attr('episode_lengths')[-1]))
                validate_episode_termination(self.training_env.buf_infos[i])
                self.logger.record('rollout/episode_termination', self.training_env.buf_infos[i]['episode_termination_type'])
                self.logger.record('memory/memory_usage_mb', psutil.Process().memory_info().rss / (1024 ** 2))

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        for env in self.training_env.envs:
            env.unwrapped.on_rollout_end()
        return

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.training_env.close()

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super().init_callback(model)
        self.is_monitor_wrapped = self.training_env.env_is_wrapped(Monitor)[0]


class SwimEvalCallback(EvalCallback):
    """
    Extension of EvalCallback for Swim Deeper agents.
    :param min_train_steps: Wait min_train_steps training steps before evaluating the model
    """

    def __init__(
            self,
            eval_env: Union[gymnasium.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            eval_inference_freq: Union[int, Tuple[int, str]] = (1, 'episode'),
            eval_freq: int = 5,
            min_train_steps: float = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, 0, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.min_train_steps = min_train_steps
        self.eval_inference_freq = convert_train_freq(eval_inference_freq)
        self.n_rollout_calls = 0
        self.continue_training = True
        self.total_eval_steps = 0

    def _init_callback(self) -> None:
        super()._init_callback()
        for env in self.eval_env.envs:
            if env.unwrapped.handler.compute_stats:
                with open(os.path.join(self.logger.dir, 'final_model_metrics.csv'), mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Algorithm', 'Seed', 'Episode', 'MeanAError', 'STDAError', 'MeanRError', 'STDRError', 'OutOfView', 'MaximumDistance', 'TargetCollision',
                                     'MeanASmoothnessError', 'STDASmoothnessError', 'MeanDSmoothnessError', 'STDDSmoothnessError'])

    def _on_step(self) -> bool:
        """
        Return continue_training as we may/may not have determined this as over
        """
        return self.continue_training

    def evaluate_policy(
            self,
            model: "type_aliases.PolicyPredictor",
            env: Union[gymnasium.Env, VecEnv],
            eval_inference_freq: TrainFreq = TrainFreq(1, TrainFrequencyUnit.EPISODE),
            deterministic: bool = False,
            render: bool = False,
            callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
            warn: bool = True,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward.
        If a vector env is passed in, this divides the episodes to evaluate onto the
        different elements of the vector env. This static division of work is done to
        remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
        details and discussion.

        .. note::
            If environment has not been wrapped with ``Monitor`` wrapper, reward and
            episode lengths are counted as it appears with ``env.step`` calls. If
            the environment contains wrappers that modify rewards or episode lengths
            (e.g. reward scaling, early episode reset), these will affect the evaluation
            results as well. You can avoid this by wrapping environment with ``Monitor``
            wrapper before anything else.

        :param model: The RL agent you want to evaluate. This can be any object
            that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
            or policy (``BasePolicy``).
        :param env: The gym environment or ``VecEnv`` environment.
        :param eval_inference_freq: Number of episode/steps to evaluate the agent
        :param deterministic: Whether to use deterministic or stochastic actions
        :param render: Whether to render the environment or not
        :param callback: callback function to do additional checks,
            called after each step. Gets locals() and globals() passed as parameters.
        :param warn: If True (default), warns user about lack of a Monitor wrapper in the
            evaluation environment.
        :return: Returns ([float], [int]), first list containing per-episode rewards and
            second containing per-episode lengths(in number of steps).
        """
        if not isinstance(env, VecEnv):
            print('[evaluate_policy] Wrapping the env in a DummyVecEnv')
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        n_envs = env.num_envs
        episode_rewards = []
        episode_lengths = []

        step_counts = np.zeros(n_envs, dtype='int')
        episode_counts = np.zeros(n_envs, dtype='int')

        count_targets = np.array([(eval_inference_freq.frequency + i) // n_envs for i in range(n_envs)], dtype="int")

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype='int')

        observations = env.reset()

        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        while should_collect_more_steps_vec(eval_inference_freq, step_counts, episode_counts, count_targets):
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic
            )

            new_observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1

            step_counts += 1
            self.total_eval_steps += 1
            for i in range(n_envs):
                if should_collect_more_steps(eval_inference_freq, step_counts[i] - 1, episode_counts[i]):
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if callback is not None:
                        callback(locals(), globals())

                    # Even if wrapped with a Monitor, we cannot use the monitor values as we supress logging for evaluation episodes
                    if dones[i] or ((eval_inference_freq.unit == TrainFrequencyUnit.STEP) and (step_counts[i] == eval_inference_freq.frequency)):
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])

                        if dones[i]:
                            validate_episode_termination(info)

                            # As with training, only log when the episode is terminated or truncated, not when steps is reached
                            self.logger.record('eval/episode_termination', info['episode_termination_type'])
                            self.logger.record('eval/ep_reward', episode_rewards[-1])
                            self.logger.record('eval/ep_length', episode_lengths[-1])
                            self.logger.record('time/total_timesteps', self.num_timesteps, exclude='tensorboard')
                            self.logger.dump(step=self.total_eval_steps)

                            if self.eval_env.envs[i].unwrapped.handler.compute_stats:
                                with open(os.path.join(self.logger.dir, 'final_model_metrics.csv'), mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([self.model.__class__.__name__,
                                                     self.eval_env.envs[i].unwrapped.seed,
                                                     self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['episode_num'],
                                                     np.mean(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['a_error']),
                                                     np.std(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['a_error']),
                                                     np.mean(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['d_error']),
                                                     np.std(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['d_error']),
                                                     self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['out_of_view'],
                                                     self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['maximum_distance'],
                                                     self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['target_collision'],
                                                     np.mean(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['a_smoothness_error']),
                                                     np.std(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['a_smoothness_error']),
                                                     np.mean(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['d_smoothness_error']),
                                                     np.std(self.eval_env.envs[i].unwrapped.handler.final_model_info_prev['d_smoothness_error'])])

                        current_rewards[i] = 0
                        current_lengths[i] = 0

                        print('[INFERENCE] Episode finished. \nReward: {:.2f} \nSteps: {}'.format(episode_rewards[-1], episode_lengths[-1]))
                        episode_counts[i] += 1

            observations = new_observations

            if render:
                env.render()

        return episode_rewards, episode_lengths

    def evaluate(self) -> None:
        """
        Method called by either an inference only run or a training run, performing evaluation metrics and reporting to logger.
        """
        print('[INFERENCE START]')

        # N.B. Only the evaluation environment needs to know about the inference, all training behaviour is self-contained
        for env in self.eval_env.envs:
            env.unwrapped.on_inference_start()

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    'Training and eval env are not wrapped the same way, '
                    'see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback '
                    'and warning above.'
                ) from e

        # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = self.evaluate_policy(
            self.model,
            self.eval_env,
            eval_inference_freq=self.eval_inference_freq,
            render=self.render,
            deterministic=self.deterministic,
            callback=self._log_success_callback,
            warn=self.warn
        )

        if self.log_path is not None:
            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps = np.append(self.evaluations_timesteps, self.num_timesteps)
            self.evaluations_results = np.append(self.evaluations_results, episode_rewards)
            self.evaluations_length = np.append(self.evaluations_length, episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )

        mean_ep_reward, std_ep_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)  # A little meaningless for continuous learning, but keep anyway

        if self.verbose >= 1:
            print(f'Eval num_timesteps={self.num_timesteps}, 'f'mean_episode_reward={mean_ep_reward:.2f} +/- {std_ep_reward:.2f}, mean episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}')

        self.logger.record('eval/mean_ep_reward', mean_ep_reward)
        self.logger.record('eval/mean_ep_length', mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f'Success rate: {100 * success_rate:.2f}%')
            self.logger.record('eval/success_rate', success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record('time/total_timesteps', self.num_timesteps, exclude='tensorboard')
        self.logger.dump(self.num_timesteps)

        if mean_ep_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print('New best mean reward!')
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
            self.best_mean_reward = float(mean_ep_reward)
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                self.continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            self.continue_training = self.continue_training and self._on_event()

        for env in self.eval_env.envs:
            env.unwrapped.on_inference_end()

    def _on_rollout_start(self) -> None:
        # We can guarantee that at this point either:
        # 1) We have just started training
        # 2) We have just complete an optimization cycle
        # We only want to evaluate in the case of 2)
        if (self.num_timesteps > self.min_train_steps) and (self.n_rollout_calls > 0) and ((self.n_rollout_calls % self.eval_freq) == 0):
            self.evaluate()
            # Now we can resume physics
            for env in self.training_env.envs:
                env.unwrapped.on_rollout_start()

        self.n_rollout_calls += 1

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self.eval_env.close()
