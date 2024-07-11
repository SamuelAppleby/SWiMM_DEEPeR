from typing import Dict, Type

from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC
}

ENVIRONMENT_TO_LOAD = 'UnderwaterEnv'
IP_HOST = '127.0.0.1'
PORT_TRAIN = 60260
PORT_INFERENCE = 60261
