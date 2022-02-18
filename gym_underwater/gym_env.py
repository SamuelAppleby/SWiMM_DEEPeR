import logging
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_underwater.sim_handler import UnitySimHandler

logger = logging.getLogger(__name__)

class UnderwaterEnv(gym.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 'Rover'
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):         

        print("starting underwater environment")

        # set logging level
        logging.basicConfig(level=logging.INFO)  

        logger.debug("DEBUG ON")

        ##########################################################################################
        # HERE IS WHERE PREVIOUSLY LAUNCHED SIM BY CREATING AN INSTANCE OF THE UNITYPROCESS CLASS 
        # AND CALLING THE START FUNCTION WHICH USED THE FUNCTUON Popen FROM THE PACKAGE subprocess 
        # WITH ARGS SIM PATH, PORT AND HOST ADDRESS. THERE WAS A SLEEP AFTER IT.
        ##########################################################################################

        #########################################################################################
        # HERE IS WHERE CREATED INSTANCE OF THE CLASS UNITYSIMCONTROLLER WHICH IN TURN CREATES
        # INSTANCE OF THE CLASS UNITYSIMHANDLER AND INSTANCE OF THE CLASS SIMCLIENT. THOUGHTS FOR 
        # NOW IS THAT A LOT OF THE FUNCTIONALITY PROVIDED BY THESE CLASSES IS MADE REDUNDANT BY
        # SERVER.PY AND CAN MAYBE COMBINE REST INTO THIS ENV CLASS. SEE IF THIS SIMPLIFIED APPROACH
        # WORKS BEFORE COPYING SAME STRUCTURE
        #########################################################################################

        self.handler = UnitySimHandler()

        # action space declaration 
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        )

        # observation space declaration           
        self.observation_space = spaces.Box(low=0, high=255, shape=(256,256,3), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = 1  

        ########################################################################################
        # WOULD BE GOOD TO FINISH INIT WITH SOME SORT OF WAIT UNTIL SIMULATION FINISHED LOADING
        # FOR EXAMPLE, A BOOL VAR LOADED, AND THEN JUST WHILE NOT LOADED: SLEEP, AND CHANGE THE
        # BOOL TO TRUE ONCE RECEIVED MESSAGE FROM SIM OR SOMETHING ALONG THOSE LINES
        ########################################################################################

    def __del__(self):
        self.close()

    def close(self):
        self.handler.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):             

        for i in range(self.frame_skip):

            # implement action in sim
            self.handler.take_action(action)                                                  

            # retrieve results of action implementation
            observation, reward, done, info = self.handler.observe()                         

            # determine if episode reached end
            if self.step_count > 3000:
                print("Episode timed out")
                done = True
            self.step_count+=1

        return observation, reward, done, info                                             

    def reset(self):

        # reset simulation to start state
        self.handler.reset()

        # reset step counter
        self.step_count = 0

        # fetch initial observation on which to base first action decision 
        observation, _, _, _ = self.handler.observe()
        
        return observation

    def render(self, mode="human", close=False):
        if close:
            self.handler.quit()
        if mode == 'rgb_array':
            return self.handler.image_array
        return self.handler.render(mode)

    def is_game_over(self):
        return self.handler.is_game_over()



