import logging
import time
import json
import os
import numpy as np 

from gym_underwater.python_server import PythonServer
from config import *

logger = logging.getLogger(__name__)

class UnitySimCommunicator:
    def __init__(self):

        logger.setLevel(logging.INFO)

        self.address = (HOST, PORT)

        self.handler = UnitySimHandler()

        self.server = PythonServer(self.address, self.handler)

    def wait_until_loaded(self):
        while self.handler.server is None:
            logger.warning("waiting for sim ..")
            time.sleep(1.0)          

    # def reset(self):
    #     self.handler.reset()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        return self.handler.observe()

    def quit(self):
        self.server.stop()

    # def render(self):
    #     pass

    # def is_game_over(self):
    #     return self.handler.is_game_over()

    # def calc_reward(self, done):
    #     return self.handler.calc_reward(done)

class UnitySimHandler():

    def __init__(self):

        self.server = None

        self.image_array = np.zeros((256,256,3))
        self.last_obs = None
        self.hit = False
        self.rover_pos = None
        self.target_pos = None
        # self.rover_fwd = None
        # self.target_fwd = None

        # self.over = False

        self.fns = {
            "on_telemetry": self.on_telemetry,
        }

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Gym ~~~~~~~~~~~~~~~~~~~~~~~~~#

    # def reset(self):

    #     logger.debug("resetting")

    #     self.image_array = np.zeros((256,256,3))
    #     self.last_obs = None
    #     self.hit = False
    #     self.rover_pos = None
    #     self.target_pos = None
    #     self.rover_fwd = None
    #     self.target_fwd = None

    #     self.over = False
    
    def observe(self):

        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array                                             
        # done = self.is_game_over()
        # reward = self.calc_reward(done)
                                              
        # info = {"dummy": "can add variables here"}                            

        # return observation, reward, done, info
        return observation

    # def calc_reward(self, done):

    #     ####### can implement reward function once wrapper working #######

    #     reward = 1

    #     return reward

    # def determine_episode_over(self):

    #     ####### implement episode termination criteria here ###########

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Socket ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def on_connect(self, server):
        logger.debug("socket connected")
        self.server = server

    def on_disconnect(self):
        logger.debug("socket disconnected")
        self.server = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def on_recv_message(self, message):

        if "msg_type" not in message:
            logger.warn("expected msg_type field")
            return
        msg_type = message["msg_type"]
        payload = message["payload"]
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](payload)
        else:
            logger.warning(f"unknown message type {msg_type}")

    def on_telemetry(self, payload):

        image = payload["jpgImage"]
        self.image_array = np.array(image)

        if SAVE_IMAGES:
            b = bytearray(image)
            self.write_image_to_file_incrementally(b)

        self.hit = payload["isColliding"]
        self.rover_pos = payload["currentPosition"]
        self.target_pos = payload["targetPositions"]
        # self.rover_fwd = payload["rover_fwd"]
        # self.target_fwd = payload["target_fwd"]   

        # if self.over:
            # return

        # self.determine_episode_over()

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def take_action(self, action):
        if self.server is None:
            return
        action_msg = {
            "msgType": "ReceiveJsonControls",
            "payload": {
                "forwardThrust": action[0].__str__(),
                "verticalThrust": action[1].__str__(),
                "yRotation": action[2].__str__(),
            }
        }
        self.server.msg = json.dumps(action_msg)

    def generate_server_config(self):
        """
        Generate server config for client
        """

        self.server.msg = json.dumps(SERVER_CONF)

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def write_image_to_file_incrementally(self, image):
        """
        Dumping the image to a continuously progressing file, just for debugging purposes
        """
        os.makedirs(SAVE_IMAGES_TO, exist_ok=True)

        i = 0
        while os.path.exists(os.path.join(SAVE_IMAGES_TO, f"sample{i}.jpeg")):
            i += 1
        with open(os.path.join(SAVE_IMAGES_TO, f"sample{i}.jpeg"), "wb") as f:
            f.write(image)

    # def is_game_over(self):
    #     return self.over






    



    
    