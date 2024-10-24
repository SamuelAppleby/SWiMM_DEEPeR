import math
from typing import Dict, Type

from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "td3": TD3,
    "ppo": PPO,
    "sac": SAC
}

ENVIRONMENT_TO_LOAD = 'UnderwaterEnv'
IP_HOST = '127.0.0.1'
PORT_TRAIN = 60260
PORT_INFERENCE = 60360


# CAMERA
SENSOR_WIDTH = 4.98  # mm
SENSOR_HEIGHT = 3.74  # mm
FOCAL_LENGTH = 2.97  # mm
CAM_FOV = math.degrees(2 * math.atan(SENSOR_WIDTH / (2 * FOCAL_LENGTH)))  # HORIZONTAL
ALPHA = CAM_FOV / 2.0


# REWARD
SMOOTHNESS_THRESHOLD = 0.5
SMOOTHNESS_PENALTY = 0.5
MAX_REWARD = 1
