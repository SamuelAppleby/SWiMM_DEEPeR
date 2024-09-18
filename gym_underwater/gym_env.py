import time
from typing import Tuple

import numpy as np

import gymnasium
from gymnasium import spaces

from gym_underwater.constants import IP_HOST, PORT_TRAIN
from gym_underwater.enums import TrainingType, ObservationType, RenderType
from gym_underwater.sim_comms import UnitySimHandler


class UnderwaterEnv(gymnasium.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle
    """

    def __init__(self,
                 obs: ObservationType,
                 img_res: Tuple[int, int, int] = (64, 64, 3),
                 cmvae=None,
                 tensorboard_log: str = None,
                 debug_logs: bool = False,
                 ip: str = IP_HOST,
                 port: int = PORT_TRAIN,
                 training_type: TrainingType = TrainingType.TRAINING,
                 render=RenderType.HUMAN,
                 seed: int = None,
                 compute_stats: bool = False):
        super().__init__()
        print('Starting underwater environment ..')

        # Make obs arg instance variable
        self.obs = obs
        self.seed = seed
        self.tensorboard_log = tensorboard_log
        self.render = render

        # action space declaration
        print('Declaring action space')
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # Create instance of class that deals with Unity communications
        self.handler = UnitySimHandler(img_res=img_res,
                                       tensorboard_log=self.tensorboard_log if debug_logs else None,
                                       ip=ip,
                                       port=port,
                                       training_type=training_type,
                                       cmvae=cmvae,
                                       action_space=self.action_space,
                                       render=self.render,
                                       seed=seed,
                                       compute_stats=compute_stats)

        self.handler.send_server_config()

        # Observation space declaration
        print('Declaring observation space')
        match self.obs:
            case ObservationType.IMAGE:
                self.observation_space = spaces.Box(low=0,
                                                    high=255,
                                                    shape=self.handler.img_res,
                                                    dtype=np.uint8)
            case ObservationType.VECTOR:
                self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                    high=np.finfo(np.float32).max,
                                                    shape=(1, 12),
                                                    dtype=np.float32)
            case ObservationType.CMVAE:
                assert cmvae is not None, 'Must provide a cmvae if declaring cmvae observation space'
                self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                    high=np.finfo(np.float32).max,
                                                    shape=(1, int(cmvae.q_img.dense2.units / 2) + (self.handler.previous_actions.maxsize * self.action_space.shape[0])),
                                                    dtype=np.float32)
            case _:
                raise ValueError(f'Invalid observation type: {obs}')

    def step(self, action):
        self.handler.send_action(action)
        return self.handler.observe(self.obs)

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))  # Seed will be present in the first call (see setup_learn), otherwise None
        self.handler.reset()
        observation, _, _, _, info = self.handler.observe(self.obs)
        return observation, info

    def on_set_world_state(self, state):
        self.handler.send_world_state(state)

    def on_rollout_start(self):
        self.handler.send_rollout_start()

    def on_rollout_end(self):
        self.handler.send_rollout_end()

    def on_inference_start(self):
        self.handler.send_inference_start()

    def on_inference_end(self):
        self.handler.send_inference_end()

    def wait_until_client_ready(self):
        while self.handler.read_write_thread.is_alive() and not self.handler.sim_ready:
            time.sleep(self.handler.interval)

    def close(self):
        self.handler.close()
