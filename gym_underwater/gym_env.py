import os
import subprocess
import threading
import time

import numpy as np

import gymnasium
from gymnasium import spaces

from gym_underwater.constants import IP_HOST, PORT_TRAIN
from gym_underwater.enums import Protocol
from gym_underwater.sim_comms import UnitySimHandler


def run_executable(path, args):
    subprocess.run([path] + args)


def launch_simulation(args, linux=False) -> threading.Thread:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'builds', 'linux', 'SWiMM_DEEPeR.x86_64') if linux else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'builds', 'windows', 'SWiMM_DEEPeR.exe')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Executable not found at {path}")

    thread = threading.Thread(target=run_executable, args=(path, args))
    thread.start()
    return thread


class UnderwaterEnv(gymnasium.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 
    """

    def __init__(self, obs, opt_d=6, max_d=4, img_res=(64, 64, 3), tensorboard_log=None, protocol=Protocol.TCP, ip=IP_HOST, port=PORT_TRAIN, seed=None, cmvae=None, exe_args=None):
        super().__init__()
        print('Starting underwater environment ..')

        # initialise VAE
        self.cmvae = cmvae
        self.z_size = int(cmvae.q_img.dense2.units / 2) if cmvae is not None else None

        # make obs arg instance variable
        self.obs = obs

        self.tensorboard_log = tensorboard_log
        self.seed = seed

        self.thread_exe = launch_simulation(args=exe_args)

        # create instance of class that deals with Unity communications
        self.handler = UnitySimHandler(opt_d=opt_d, max_d=max_d, img_res=img_res, tensorboard_log=tensorboard_log, protocol=protocol, ip=ip, port=port, seed=seed)
        self.handler.send_server_config()

        # action space declaration
        print('Declaring action space')
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # observation space declaration
        print('Declaring observation space')
        if self.obs == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=self.handler.img_res, dtype=np.uint8)
        elif self.obs == 'vector':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, 12), dtype=np.float32)
        elif self.obs == 'cmvae':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, self.z_size), dtype=np.float32)
        else:
            raise ValueError(f'Invalid observation type: {obs}')

    def observe_and_process_observation(self):
        # retrieve results of action implementation
        observation, reward, terminated, truncated, info = self.handler.observe(self.obs)
        # if vae has been passed, raw image observation encoded to latent vector
        if self.cmvae is not None:
            # add a dimension on the front so that has the shape (N, vae_res, vae_res, 3) that network expects
            observation = np.expand_dims(observation, axis=0)

            # set latent vector as observation
            observation, _, _ = self.cmvae.encode(observation)

        return observation, reward, terminated, truncated, info

    def step(self, action):
        self.handler.send_action(action)
        return self.observe_and_process_observation()

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))  # Seed will be present in the first call (see setup_learn), otherwise None
        self.handler.reset()
        observation, _, _, _, info = self.observe_and_process_observation()
        return observation, info

    def render(self):
        return self.handler.image_array

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
        self.thread_exe.join()
