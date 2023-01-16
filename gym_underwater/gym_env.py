import logging
import warnings
import gym
import numpy as np
from gym import spaces
from sim_comms import UnitySimHandler

warnings.filterwarnings("ignore", category=UserWarning, module='gym')
logger = logging.getLogger(__name__)


class UnderwaterEnv(gym.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 
    """

    def __init__(self, obs, opt_d, max_d, scale):
        print("Starting underwater environment ..")

        # set logging level
        logging.basicConfig(level=logging.INFO)
        logger.debug("DEBUG ON")

        # make obs arg instance variable
        self.obs = obs
        self.scale = scale

        # create instance of class that deals with Unity comms
        self.handler = UnitySimHandler(opt_d, max_d, scale)

        # action space declaration
        print("Declaring action space")
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # observation space declaration
        print("Declaring observation space")
        if self.obs == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=self.scale, dtype=np.uint8)
        elif self.obs == 'vector':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, 12), dtype=np.float32)
        else:
            raise ValueError('Invalid observation type: {}'.format(obs))

        #     # seed environment
        #     #self.seed()

        # wait until connection established 
        self.handler.wait_until_loaded()
        self.handler.send_server_config()
        self.handler.wait_until_training_ready()

    def close(self):
        self.handler.quit()

    # #def seed(self, seed=None):
    #     #self.np_random, seed = seeding.np_random(seed)
    #     #return [seed]

    def step(self, action):
        # send action decision to handler to send off to sim
        self.handler.send_action(action)

        # retrieve results of action implementation
        observation, reward, done, info = self.handler.observe(self.obs)

        return observation, reward, done, info

    def reset(self):
        # reset simulation to start state
        self.handler.reset()

        # fetch initial observation
        observation, reward, done, info = self.handler.observe(self.obs)

        return observation

    def render(self):
        return self.handler.handler.image_array
