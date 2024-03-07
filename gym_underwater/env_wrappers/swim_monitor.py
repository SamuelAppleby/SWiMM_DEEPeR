from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from stable_baselines3.common.monitor import Monitor

from gym_underwater.enums import TrainingType


class SwimMonitor(Monitor):
    """
    A subclassed monitor wrapper for UnderwaterEnv environments, it enhances the base model by ignoring inference metrics.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env, filename=filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords, info_keywords=info_keywords, override_existing=override_existing)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.env.unwrapped.handler.training_type == TrainingType.TRAINING:
            return super().step(action)

        return self.env.step(action)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        if self.env.unwrapped.handler.training_type == TrainingType.TRAINING:
            return super().reset(**kwargs)

        return self.env.reset(**kwargs)
