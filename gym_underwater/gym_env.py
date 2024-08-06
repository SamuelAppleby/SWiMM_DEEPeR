import time
from queue import Queue

import numpy as np

import gymnasium
from gymnasium import spaces

from gym_underwater.constants import IP_HOST, PORT_TRAIN
from gym_underwater.enums import TrainingType
from gym_underwater.mathematics import normalized_absolute_difference
from gym_underwater.sim_comms import UnitySimHandler


class UnderwaterEnv(gymnasium.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle
    """

    def __init__(self, obs, img_res=(64, 64, 3), cmvae=None, tensorboard_log=None, debug_logs=False, ip=IP_HOST, port=PORT_TRAIN, training_type=TrainingType.TRAINING, seed=None):
        super().__init__()
        print('Starting underwater environment ..')

        # initialise VAE
        self.cmvae = cmvae
        self.z_size = int(cmvae.q_img.dense2.units / 2) if cmvae is not None else None

        # make obs arg instance variable
        self.obs = obs

        self.seed = seed

        self.tensorboard_log = tensorboard_log

        # create instance of class that deals with Unity communications
        self.handler = UnitySimHandler(img_res=img_res, tensorboard_log=self.tensorboard_log if debug_logs else None, ip=ip, port=port, training_type=training_type, seed=seed)
        self.handler.send_server_config()

        # action space declaration
        print('Declaring action space')
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        self.previous_actions = Queue(maxsize=10)

        # observation space declaration
        print('Declaring observation space')
        if self.obs == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=self.handler.img_res, dtype=np.uint8)
        elif self.obs == 'vector':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, 12), dtype=np.float32)
        elif self.obs == 'cmvae':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, self.z_size + (self.previous_actions.maxsize * self.action_space.shape[0])), dtype=np.float32)
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

        smoothness_penalty = 0
        if self.previous_actions.qsize() > 1:
            action_list = list(self.previous_actions.queue)
            action_diff = normalized_absolute_difference(action_list[-1], action_list[-2], self.action_space)

            for diff in action_diff:
                if diff > 0.25:
                    smoothness_penalty += 1

        return observation.numpy(), reward, terminated, truncated, info

    def step(self, action):
        self.handler.send_action(action)
        observation, reward, terminated, truncated, info = self.observe_and_process_observation()

        if self.previous_actions.full():
            self.previous_actions.get()  # Remove the oldest action
        self.previous_actions.put(action)  # Add the new action

        observation = self.get_augmented_state(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get('seed'))  # Seed will be present in the first call (see setup_learn), otherwise None
        self.handler.reset()

        self.previous_actions.queue.clear()

        observation, _, _, _, info = self.observe_and_process_observation()
        observation = self.get_augmented_state(observation)
        return observation, info

    def get_augmented_state(self, state):
        # Flatten the list of previous actions
        previous_actions_np = np.array(list(self.previous_actions.queue)).flatten()

        if len(previous_actions_np) < 20:
            padding = np.zeros(20 - len(previous_actions_np))
            previous_actions_np = np.concatenate([previous_actions_np, padding])

        return np.concatenate([state.flatten(), previous_actions_np]).reshape(1, -1)

    def render(self):
        pass

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
