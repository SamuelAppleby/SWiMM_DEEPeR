import logging
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_underwater.sim_comms import UnitySimCommunicator 

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

    #     # action space declaration 
    #     self.action_space = spaces.Box(
    #         low=np.array([-1, -1, -1]),
    #         high=np.array([1, 1, 1]),
    #         dtype=np.float32,
    #     )

    #     # observation space declaration           
    #     self.observation_space = spaces.Box(low=0, high=255, shape=(256,256,3), dtype=np.uint8)

    #     # seed environment
    #     #self.seed()

    #     # wait for sim to load
    #     #self.communicator.wait_until_loaded()

    # #def close(self):
    #     #self.communicator.quit()

    # #def seed(self, seed=None):
    #     #self.np_random, seed = seeding.np_random(seed)
    #     #return [seed]

    # def step(self, action):             

    #     # send action decision to communicator to send off to sim
    #     self.communicator.take_action(action)                                                  

    #     # retrieve results of action implementation
    #     observation, reward, done, info = self.communicator.observe()                         

    #     return observation, reward, done, info                                             

    # def reset(self):

    #     # reset simulation to start state
    #     self.communicator.reset()

    #     # fetch initial observation  
    #     observation, _, _, _ = self.communicator.observe()
        
    #     return observation

    # def render(self):
    #     return self.communicator.image_array

    # def is_game_over(self):
    #     return self.communicator.is_game_over()







