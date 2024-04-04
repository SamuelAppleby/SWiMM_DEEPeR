import logging
import time

import numpy as np
import cv2

import cmvae_utils.dataset_utils

import gymnasium
from gymnasium import spaces

from gym_underwater.sim_comms import UnitySimHandler


class UnderwaterEnv(gymnasium.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 
    """

    def __init__(self, cmvae, obs, opt_d, max_d, img_res, tensorboard_log, protocol, host, seed):
        super().__init__()
        print('Starting underwater environment ..')

        # set logging level
        logging.basicConfig(level=logging.INFO)

        # initialise VAE
        self.cmvae = cmvae
        self.z_size = int(cmvae.q_img.dense2.units / 2) if cmvae is not None else None

        # make obs arg instance variable
        self.obs = obs

        # create instance of class that deals with Unity communications
        self.handler = UnitySimHandler(opt_d, max_d, img_res, tensorboard_log, protocol, host, seed)

        self.handler.connect(*self.handler.address)
        self.handler.read_write_thread.start()

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

    def observe_and_process_observation(self, action=None, pred=False):
        # retrieve results of action implementation
        observation, reward, terminated, truncated, info = self.handler.observe(self.obs)
        # if vae has been passed, raw image observation encoded to latent vector
        if self.cmvae is not None:
            # vae will have been trained on BGR ordered image arrays, so need to reverse first and last channel of RGB array
            observation = observation[:, :, ::-1]
            # resize to resolution used to train vae
            observation = cv2.resize(observation, (self.handler.img_res[0], self.handler.img_res[1]))
            # normalize pixel values
            observation = observation / 255.0 * 2.0 - 1.0
            # add a dimension on the front so that has the shape (?, vae_res, vae_res, 3) that network expects
            observation = observation.reshape(-1, *observation.shape)
            # pass through encoder network
            if pred:
                _, _, z, pred = self.cmvae.encode_with_pred(observation)
                # denormalize state predictions
                pred = cmvae_utils.dataset_utils.de_normalize_gate(pred)
                print(f'Prediction: {pred[0]}, Thrust: {action[0]}, Steer: {action[1]}')
            else:
                _, _, z = self.cmvae.encode(observation)

            # set latent vector as observation
            observation = z

        return observation, reward, terminated, truncated, info

    def step(self, action):
        self.handler.send_action(action)
        return self.observe_and_process_observation()

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))      # Seed will be present in the first call (see setup_learn), otherwise None
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

    def on_inference_start(self, eval_inference_freq):
        self.handler.send_inference_start(eval_inference_freq)

    def on_inference_end(self):
        self.handler.send_inference_end()

    def wait_until_client_ready(self):
        while self.handler.read_write_thread.is_alive() and not self.handler.sim_ready:
            time.sleep(self.handler.interval)

    def close(self):
        self.handler.close()
