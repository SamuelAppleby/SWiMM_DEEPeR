"""Wrapper for limiting the time steps of an environment."""
from gymnasium.wrappers import TimeLimit
import gymnasium as gym

from gym_underwater.enums import TrainingType


class SwimTimeLimit(TimeLimit):
    """This wrapper extends TimeLimit by providing early termination functionality for both training and inference phases.
    It also prevents the truncation of episodes that are already marked for termination
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps_train: int = 3000,
        max_episode_steps_inference: int = 3000
    ):
        """Initializes the :class:`SwimTimeLimit` wrapper with an environment and the number of steps after which truncation will occur
        for both during training and inferring.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps_train: Max episode steps for training (if ``None``, 3000 is used)
            max_episode_steps_inference: Max episode steps for inference (if ``None``, 3000 is used)
        """
        super().__init__(env, max_episode_steps_train)
        self.max_episode_steps_inference = max_episode_steps_inference
        self._elapsed_steps_inference = 0

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        if self.env.unwrapped.handler.training_type == TrainingType.TRAINING:
            observation, reward, terminated, truncated, info = super().step(action)
            if truncated:
                if terminated:
                    truncated = False
                else:
                    print(f'[EPISODE TERMINATED] Truncating as current steps {self._elapsed_steps} is at the limit {self._max_episode_steps}')
            return observation, reward, terminated, truncated, info

        observation, reward, terminated, truncated, info = self.env.step(action)

        self._elapsed_steps_inference += 1
        if (self._elapsed_steps_inference >= self.max_episode_steps_inference) and not terminated:
            print(f'[EPISODE TERMINATED] Truncating as current steps {self._elapsed_steps_inference} is at the limit {self.max_episode_steps_inference}')
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        if self.env.unwrapped.handler.training_type == TrainingType.TRAINING:
            return super().reset(**kwargs)

        self._elapsed_steps_inference = 0
        return self.env.reset(**kwargs)
