from stable_baselines3.common import base_class
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os
from typing import Union, Optional, Tuple
import numpy as np
from tqdm import tqdm

import gymnasium

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from .env_wrappers.swim_monitor import SwimMonitor
from .env_wrappers.swim_time_limit import SwimTimeLimit
from .sim_comms import MAX_STEP_REWARD
from .enums import EpisodeTerminationType
from .utils.utils import evaluate_policy, convert_train_freq


class SwimCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.is_monitor_wrapped = False

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        for _format in self.logger.output_formats:
            if isinstance(_format, TensorBoardOutputFormat):
                with open('tensorboard_dir.txt', 'w') as log_file:
                    print('Tensorboard event file: {}'.format(_format.writer.file_writer.event_writer._file_name))
                    log_file.write('Tensorboard event file: {}\n'.format(_format.writer.file_writer.event_writer._file_name))

        return

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.num_timesteps > 0:
            self.logger.dump(self.num_timesteps)

        self.training_env.envs[0].unwrapped.on_rollout_start()
        return

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.training_env.buf_dones[0]:
            if self.is_monitor_wrapped:
                print('[TRAINING] Episode finished. \nReward: {:.2f} \nSteps: {}'.format(self.training_env.envs[0].get_wrapper_attr('episode_returns')[-1], self.training_env.envs[0].get_wrapper_attr('episode_lengths')[-1]))
            if self.training_env.buf_infos[0]['episode_termination_type'] is None:
                assert self.training_env.buf_infos[0]["TimeLimit.truncated"]
                self.training_env.buf_infos[0]['episode_termination_type'] = EpisodeTerminationType.THRESHOLD_REACHED

            self.logger.record('rollout/episode_termination', self.training_env.buf_infos[0]['episode_termination_type'])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.training_env.envs[0].unwrapped.on_rollout_end()
        return

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super().init_callback(model)
        self.is_monitor_wrapped = self.training_env.env_is_wrapped(SwimMonitor)[0]


class SwimEvalCallback(EvalCallback):
    """
    Extension of EvalCallback for Swim Deeper agents.
    stable_baselines3.common.callbacks.StopTrainingOnRewardThreshold.reward_threshold is now treated as a fraction of the total episode/step reward
    """

    def __init__(
            self,
            eval_env: Union[gymnasium.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            eval_inference_freq: Union[int, Tuple[int, str]] = (1, 'episode'),
            eval_freq: int = 5,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, 0, eval_freq, log_path, best_model_save_path, deterministic, render, verbose, warn)
        self.is_time_limit_wrapped = False
        self.eval_inference_freq = convert_train_freq(eval_inference_freq)
        self.n_rollout_calls = 0
        self.continue_training = True

    def _on_step(self) -> bool:
        """
        Return continue_training as we may/may not have determined this as over
        """
        return self.continue_training

    def evaluate(self, inference_only: bool = False) -> None:
        """
        Method called by either an inference only run or a training run, performing evaluation metrics and reporting to logger.
        """
        # We could have either just reset or stepped, so cache the relevant data
        if not inference_only:
            restore_state = self.training_env.reset_infos[0] if self.training_env.buf_dones[0] else self.training_env.buf_infos[0]
            self.training_env.envs[0].unwrapped.on_inference_start(self.eval_inference_freq)

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
        else:
            self.eval_env.envs[0].unwrapped.on_inference_start(self.eval_inference_freq)

        # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
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
        # Add to current Logger
        self.logger.record('eval/mean_ep_reward', float(mean_ep_reward))
        self.logger.record('eval/mean_ep_length', mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f'Success rate: {100 * success_rate:.2f}%')
            self.logger.record('eval/success_rate', success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record('time/total_timesteps', self.num_timesteps, exclude='tensorboard')
        self.logger.dump(self.num_timesteps)

        if not inference_only:
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

            self.training_env.envs[0].unwrapped.on_set_world_state({
                'rover': restore_state['rover'],
                'target': restore_state['target']
            })

            self.training_env.envs[0].unwrapped.on_inference_end()
        else:
            self.eval_env.envs[0].unwrapped.on_inference_end()

    def _on_rollout_start(self) -> None:
        # We can guarantee that at this point either:
        # 1) We have just started training
        # 2) We have just complete an optimization cycle
        # We only want to evaluate in the case of 2)
        self.n_rollout_calls += 1
        if (self.n_rollout_calls > 1) and (((self.n_rollout_calls - 1) % self.eval_freq) == 0):
            self.evaluate(inference_only=False)

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super().init_callback(model)
        self.is_time_limit_wrapped = self.training_env.env_is_wrapped(SwimTimeLimit)[0]

        if isinstance(self.callback_on_new_best, StopTrainingOnRewardThreshold):
            self.callback_on_new_best.reward_threshold *= MAX_STEP_REWARD * self.eval_env.envs[0].get_wrapper_attr('max_episode_steps_inference')
            if self.eval_inference_freq.unit == TrainFrequencyUnit.EPISODE:
                assert self.is_time_limit_wrapped, (
                    'If providing a StopTrainingOnRewardThreshold on episodic learning, you must wrap the environment in a SwimTimeLimit wrapper.'
                )


class SwimProgressBarCallback(BaseCallback):
    """
    base ProgressBarCallback doesn't behave as intended, so use the same custom implementation from SB3
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals['total_timesteps'] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()
