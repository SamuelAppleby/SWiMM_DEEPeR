import csv
import os
import time
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean


class SwimCallback(BaseCallback):
    def __init__(self, verbose=0):
        # if algo != 'sac':
        #     raise NotImplementedError("Callback creation not implemented yet for {}".format(algo))

        super().__init__(verbose)
        self.best_mean_reward = -np.inf

    def _on_training_start(self) -> None:
        self.best_mean_reward_log = os.path.join(self.locals['self'].tensorboard_log, 'best_episode_rewards.csv')

        with open(self.best_mean_reward_log, 'w', newline='') as csv_file:
            csv.writer(csv_file).writerow(['Current Episode', 'Mean Reward Over Last {} Episodes'.format(self.model._stats_window_size)])
            csv_file.close()

        return

    def _on_rollout_end(self) -> None:
        print('Episode finished. \nReward: {:.2f} \nSteps: {}'.format(self.training_env.envs[0].episode_returns[-1], self.training_env.envs[0].episode_lengths[-1]))

        current_mean_rew = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if current_mean_rew > self.best_mean_reward:
            self.best_mean_reward = current_mean_rew
            begin_time = time.time()
            print('Saving best model ...')
            self.locals['self'].save(os.path.join(self.locals['self'].tensorboard_log, 'best_model'))

            with open(self.best_mean_reward_log, 'a', newline='') as csv_file:
                csv.writer(csv_file).writerow([self.model._episode_num, self.best_mean_reward])
                csv_file.close()

            print('Model saved, time taken: ', time.time() - begin_time)

        if self.model._episode_num == 1:
            for _format in self.logger.output_formats:
                if isinstance(_format, TensorBoardOutputFormat):
                    with open('train_output.txt', 'a') as log_file:
                        print('Tensorboard event file: {}'.format(_format.writer.file_writer.event_writer._file_name))
                        log_file.write('Tensorboard event file: {}\n'.format(_format.writer.file_writer.event_writer._file_name))
        return

    def _on_step(self) -> bool:
        if (self.locals['num_collected_steps'] % 100) == 0:
            print('Current Episode Steps: {}'.format(self.locals['num_collected_steps']))

        if self.training_env.buf_dones[0]:
            self.logger.record("rollout/episode_termination", self.training_env.envs[0].handler.episode_termination_type)

        return True
