import logging
import time
import warnings
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_underwater.sim_comms import UnitySimCommunicator
from config import IMG_SCALE

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

logger = logging.getLogger(__name__)


class UnderwaterEnv(gym.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 
    """

    def __init__(self):
        print("Starting underwater environment ..")

        # set logging level
        logging.basicConfig(level=logging.INFO)
        logger.debug("DEBUG ON")

        # create instance of class that deals with Unity comms
        self.communicator = UnitySimCommunicator()

        # action space declaration
        print("Declaring action space")
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # observation space declaration
        print("Declaring observation space")
        #self.observation_space = spaces.Box(low=0, high=255, shape=IMG_SCALE, dtype=np.uint8)
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1,12), dtype=np.float32)

        #     # seed environment
        #     #self.seed()

        # wait until connection established 
        self.communicator.wait_until_loaded()
        self.communicator.send_server_config()
        self.communicator.wait_until_training_ready()

    def close(self):
        self.communicator.quit()

    # #def seed(self, seed=None):
    #     #self.np_random, seed = seeding.np_random(seed)
    #     #return [seed]

    def step(self, action):

        # send action decision to communicator to send off to sim
        self.communicator.take_action(action)

        # retrieve results of action implementation
        observation, reward, done, info = self.communicator.observe()

        return observation, reward, done, info

    def reset(self):
        # reset simulation to start state
        self.communicator.reset()

        # fetch initial observation
        observation, reward, done, info = self.communicator.observe()

        return observation

    def render(self):
        return self.communicator.handler.image_array
