from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from stable_baselines3.common.monitor import Monitor

from gym_underwater.enums import TrainingType


class SwimMonitor(Monitor):
    """
    A subclassed monitor wrapper for UnderwaterEnv environments, it enhances the base model by ignoring inference metrics.
    """

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
        inference_only: bool = False
    ):
        super().__init__(env=env, filename=filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords, info_keywords=info_keywords, override_existing=override_existing)
        self.inference_only = inference_only

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if (self.env.unwrapped.handler.training_type == TrainingType.TRAINING) or self.inference_only:
            return super().step(action)

        return self.env.step(action)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        if (self.env.unwrapped.handler.training_type == TrainingType.TRAINING) or self.inference_only:
            return super().reset(**kwargs)

        return self.env.reset(**kwargs)
