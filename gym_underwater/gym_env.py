# generic imports
import logging
import warnings
import sys
import os
import numpy as np
import cv2

# specialised imports
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# local imports
from gym_underwater.sim_comms import UnitySimHandler
import cmvae_utils

# remove warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
logger = logging.getLogger(__name__)


class UnderwaterEnv(gymnasium.Env):
    """
    OpenAI Gym Environment for controlling an underwater vehicle 
    """

    def __init__(self, cmvae, obs, opt_d, max_d, scale, debug, protocol, host, ep_len_threshold):
        print("Starting underwater environment ..")

        # set logging level
        logging.basicConfig(level=logging.INFO)
        logger.debug("DEBUG ON")

        # initialise VAE
        self.cmvae = cmvae
        self.z_size = None
        if cmvae is not None:
            self.z_size = cmvae.z_size

        # make obs arg instance variable
        self.obs = obs
        self.scale = scale

        self.episode_num = 0
        self.step_num = 0

        # create instance of class that deals with Unity comms
        self.handler = UnitySimHandler(opt_d, max_d, scale, debug, protocol, host, ep_len_threshold)

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
        elif self.obs == 'vae':
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, shape=(1, self.z_size), dtype=np.float32)
        else:
            raise ValueError('Invalid observation type: {}'.format(obs))

        # wait until connection established 
        self.handler.wait_until_loaded()
        self.handler.send_server_config()
        self.handler.wait_until_client_ready()

    def close(self):
        self.handler.quit()

    # calls a Gym utility function which generates a random number generator from the seed and returns the generator and seed
    # although stable baselines utility function seeds tf, np and random, I think this just ensures nothing is missed if env uses anything else
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observe_and_process_observation(self, action=None, pred=False):
        # retrieve results of action implementation
        observation, reward, terminated, truncated, info = self.handler.observe(self.obs)

        # if vae has been passed, raw image observation encoded to latent vector
        if self.cmvae is not None:
            # vae will have been trained on BGR ordered image arrays, so need to reverse first and last channel of RGB array
            observation = observation[:, :, ::-1]
            # resize to resolution used to train vae
            observation = cv2.resize(observation, (self.cmvae.res, self.cmvae.res))
            # normalize pixel values
            observation = observation / 255.0 * 2.0 - 1.0
            # add a dimension on the front so that has the shape (?, vae_res, vae_res, 3) that network expects
            observation = observation.reshape(-1, *observation.shape)
            # pass through encoder network
            if pred:
                _, _, z, pred = self.cmvae.encode_with_pred(observation)
                # denormalize state predictions
                pred = cmvae_utils.dataset_utils.de_normalize_state(pred)
                print("Distance: {}, Prediction: {}, Thrust: {}, Steer: {}".format(self.handler.raw_d, pred[0], action[0], action[1]))
            else:
                _, _, z = self.cmvae.encode(observation)

            # set latent vector as observation
            observation = z

        return observation, reward, terminated, truncated, info

    def step(self, action):
        self.step_num += 1

        # send action decision to handler to send off to sim
        self.handler.send_action(action, self.step_num)
        return self.observe_and_process_observation()

    def reset(self, **kwargs):
        self.step_num = 0
        self.episode_num += 1

        # reset simulation to start state
        self.handler.send_reset(self.episode_num)
        # fetch initial observation
        observation, _, _, _, info = self.observe_and_process_observation()
        return observation, info

    def render(self):
        return self.handler.image_array
