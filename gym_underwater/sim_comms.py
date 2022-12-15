import base64
import logging
import time
import json
import os
import numpy as np
import math
from io import BytesIO
from PIL import Image
from skimage.transform import resize

from gym_underwater.python_server import PythonServer
from Configs.config import *

logger = logging.getLogger(__name__)


class UnitySimCommunicator:
    def __init__(self):
        logger.setLevel(logging.INFO)
        self.handler = UnitySimHandler()

    def wait_until_loaded(self):
        while not self.handler.server_connected:
            logger.warning("waiting for sim ..")
            time.sleep(1.0)

    def wait_until_training_ready(self):
        while not self.handler.sim_training_ready:
            logger.warning("waiting for sim trainig message ..")
            time.sleep(1.0)

    def send_server_config(self):
        self.handler.send_server_config()

    def reset(self):
        self.handler.reset()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self, obs):
        return self.handler.observe(obs)

    def quit(self):
        self.handler.server.stop()

    def render(self):
        pass

    def calc_reward(self):
        return self.handler.calc_reward()


class UnitySimHandler:

    def __init__(self):
        self.sim_training_ready = False
        self.server_connected = False
        self.server = PythonServer(self)
        self.image_array = np.zeros((256, 256, 3))
        self.last_obs = self.image_array
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.d = 0.0
        self.a = 0.0

        self.fns = {
            "connection_request": self.connection_request,
            'server_config_received': self.server_config_confirmation,
            'training_ready': self.sim_training_ready_request,
            "on_telemetry": self.on_telemetry
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Gym ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def reset(self):
        logger.debug("resetting")

        self.send_reset()

        self.image_array = np.zeros((256, 256, 3))
        self.last_obs = self.image_array
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.d = 0.0
        self.a = 0.0

    def observe(self, obs):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array

        # for vector obs training run, overwrite image observation with vector obs 
        # observation and self.last_obs left in because orchestrates above while loop which is making Python server wait for next message from client
        if obs == 'vector':
            observation = [self.rover_pos[0], self.rover_pos[1], self.rover_pos[2], self.rover_fwd[0], self.rover_fwd[1], self.rover_fwd[2],
                            self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_fwd[0], self.target_fwd[1], self.target_fwd[2]]

        reward = self.calc_reward()

        done = self.determine_episode_over()

        info = {"rov_pos": self.rover_pos, "targ_pos": self.target_pos, "dist": self.d, "raw_dist": self.raw_d, "rov_fwd": self.rover_fwd, "targ_fwd": self.target_fwd, "ang_error": self.a}

        return observation, reward, done, info

    def calc_reward(self):
        # heading vector from rover to target
        heading = self.target_pos - self.rover_pos

        # normalize
        norm_heading = heading / np.linalg.norm(heading)

        # if target is ahead of rover, heading[2] (i.e z-coord) should be positive and vice versa
        # so that the optimal tracking position dictated by heading[2] - OPT_D is always *behind* the target
        # regardless of travelling direction in world
        if np.dot(norm_heading, self.target_fwd) > 0:
            heading[2] = math.fabs(heading[2])
        else:
            heading[2] = -math.fabs(heading[2])

        self.raw_d = math.sqrt(math.pow(heading[0], 2) + math.pow(heading[2], 2))

        # calculate distance i.e. magnitude of heading vector
        # NOTE THAT THIS IS DISTANCE FROM ROVER TO OPTIMAL TRACKING POSITION, NOT ROVER TO TARGET
        self.d = math.sqrt(math.pow(heading[0], 2) + math.pow((heading[2] - OPT_D), 2))

        # calculate angle between rover's forward facing vector and heading vector
        self.a = math.degrees(math.atan2(norm_heading[0], norm_heading[2]) - math.atan2(self.rover_fwd[0], self.rover_fwd[2]))

        # scaling function taken from Luo et al. (2018), range [-1, 1], distance and angle equal contribution
        reward = 1.0 - ((self.d / MAX_D) + (math.fabs(self.a) / 180))

        return reward

    def determine_episode_over(self):
        if self.d > MAX_D:
            print("Episode terminated as target out of range {}".format(self.d))
            logger.debug(f"game over: distance {self.d}")
            return True
        if "Dolphin" in self.hit:
            print("Episode terminated due to collision")
            logger.debug(f"game over: hit {self.hit}")
            return True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Socket ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def on_connect(self):
        logger.debug("socket connected")
        self.server_connected = True

    def on_disconnect(self):
        logger.debug("socket disconnected")
        self.server = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def on_recv_message(self, message):

        if "msg_type" not in message:
            logger.warning("expected msg_type field")
            return
        msg_type = message["msg_type"]
        payload = message["payload"]
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](payload)
        else:
            logger.warning(f"unknown message type {msg_type}")

    def connection_request(self, payload):
        # let handler know when connection has been made
        self.on_connect()
        return

    def server_config_confirmation(self, payload):
        self.send_awaiting_training()
        return

    def sim_training_ready_request(self, payload):
        logger.debug('sim ready to train')
        self.sim_training_ready = True

    def on_telemetry(self, payload):
        self.rover_pos = np.array([payload['position'][0], payload['position'][1], payload['position'][2]])
        self.hit = payload['collision_objects']
        self.rover_fwd = np.array([payload["fwd"][0], payload['fwd'][1], payload['fwd'][2]])

        # TODO: implement receiving data on multiple targets
        # Sam.A, targets are now an array, use the last element of it for targeting
        for target in payload['targets']:
            self.target_pos = np.array([target['position'][0], target['position'][1], target['position'][2]])
            self.target_fwd = np.array([target['fwd'][0], target['fwd'][1], target['fwd'][2]])

        image = bytearray(base64.b64decode(payload['jpg_image']))

        if 'image_dir' in self.server.debug_config:
            self.write_image_to_file_incrementally(image)

        image = np.array(Image.open(BytesIO(image)))

        # scale image otherwise model optimisation takes approx 15 minutes
        image = resize(image, IMG_SCALE, 0)

        self.image_array = image

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def take_action(self, action):
        if self.server is None:
            return

        self.server.msg = {
            'msgType': 'receive_json_controls',
            'payload': {
                'jsonControls': {
                    'swayThrust': '0',
                    'heaveThrust': '0',
                    'surgeThrust': action[0].__str__(),
                    'pitchThurst': '0',
                    'yawThrust': action[1].__str__(),
                    'rollThrust': '0',
                    'depthHoldMode': '1'
                }
            }
        }

    def send_server_config(self):
        """
        Generate server config for client
        """
        self.server.msg = self.server.server_config

    def send_reset(self):
        self.server.msg = {
            'msgType': 'reset_episode',
            'payload': {}
        }

    def send_awaiting_training(self):
        self.server.msg = {
            'msgType': 'awaiting_training',
            'payload': {}
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def write_image_to_file_incrementally(self, image):
        """
        Dumping the image to a continuously progressing file, just for debugging purposes
        """
        os.makedirs(self.server.debug_config['image_dir'], exist_ok=True)

        i = 0
        while os.path.exists(os.path.join(self.server.debug_config['image_dir'], f'image{i}.jpg')):
            i += 1
        with open(os.path.join(self.server.debug_config['image_dir'], f'image{i}.jpg'), 'wb') as f:
            f.write(image)
