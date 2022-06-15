import logging
import time
import json
import os
import numpy as np 
import math

from gym_underwater.python_server import PythonServer
from config import *

logger = logging.getLogger(__name__)

class UnitySimCommunicator:
    def __init__(self):

        logger.setLevel(logging.INFO)

        self.handler = UnitySimHandler()

        self.server = PythonServer(self.handler)


    def wait_until_loaded(self):
        while self.handler.server is None:
            logger.warning("waiting for sim ..")
            time.sleep(1.0)          

    def reset(self):
        self.handler.reset()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        return self.handler.observe()

    def quit(self):
        self.server.stop()

    def render(self):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)

class UnitySimHandler():

    def __init__(self):

        self.server = None

        self.image_array = np.zeros((256,256,3))
        self.last_obs = self.image_array
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)

        self.over = False

        self.d = 0.0
        # self.a = 0.0

        self.fns = {
            "on_telemetry": self.on_telemetry,
        }

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Gym ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def reset(self):

        logger.debug("resetting")

        self.send_reset()
        # Sam.A Talk to Kirsten about this
        #time.sleep(1)

        self.image_array = np.zeros((256,256,3))
        self.last_obs = self.image_array
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)

        self.over = False
    
    def observe(self):

        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array                                             
        done = self.is_game_over()
        reward = self.calc_reward(done)                                   
        info = {"dummy": "can add variables here"}                            

        return observation, reward, done, info

    def calc_reward(self, done):

        if done:
            return 0

        # heading vector from rover to target
        heading = self.target_pos - self.rover_pos

        # normalize
        # norm_heading = heading/np.linalg.norm(heading)

        # if target is ahead of rover, heading[2] (i.e z-coord) should be positive and vice versa
        # so that the optimal tracking position dictated by heading[2] - OPT_D is always *behind* the target
        # regardless of travelling direction in world
        # if np.dot(norm_heading, self.target_fwd) > 0:
            # heading[2] = math.fabs(heading[2])
        # else:
            # heading[2] = -math.fabs(heading[2])

        # calculate distance i.e. magnitude of heading vector
        # NOTE THAT THIS IS DISTANCE FROM ROVER TO OPTIMAL TRACKING POSITION, NOT ROVER TO TARGET
        self.d = math.sqrt(math.pow(heading[0], 2) + math.pow((heading[2] - OPT_D), 2))

        # calculate angle between rover's forward facing vector and heading vector
        # self.a = math.degrees(math.atan2(norm_heading[0], norm_heading[2]) - math.atan2(self.rover_fwd[0], self.rover_fwd[2]))

        # scaling function taken from Luo et al. (2018), range [-1, 1], distance and angle equal contribution
        reward = 1.0 - self.d / MAX_D
        # reward = 1.0 - ((self.d / MAX_D) + (math.fabs(self.a) / 180))

        return reward

    def determine_episode_over(self):
        #if self.d > MAX_D:
            #logger.debug(f"game over: distance {self.d}")
            #self.over = True
            #print("Episode terminated as target out of range")
        if "Fish" in self.hit:
            logger.debug(f"game over: hit {self.hit}")
            #self.over = True
            print("Episode terminated due to collision")

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

        image = payload["jpg_image"]
        self.image_array = np.array(image)

        if self.server.debug_config["save_images"]:
            b = bytearray(image)
            self.write_image_to_file_incrementally(b)

        self.rover_pos = np.array([payload["position"]["x"], payload["position"]["y"], payload["position"]["z"]])
        self.hit = payload["collision_objects"]
        self.rover_fwd = np.array([payload["fwd"]["x"], payload["fwd"]["y"], payload["fwd"]["z"]])

        # TODO: implement receiving data on multiple targets
        # Sam.A, targets are now an array, use the last element of it for targeting
        for target in payload["targets"]:
            self.target_pos = np.array([target["position"]["x"], target["position"]["y"], target["position"]["z"]])
            self.target_fwd = np.array([target["fwd"]["x"], target["fwd"]["y"], target["fwd"]["z"]])

        if self.over:
            return

        self.determine_episode_over()

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def take_action(self, action):
        if self.server is None:
            return
        action_msg = {
            "msgType": "receive_json_controls",
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
        self.server.msg = json.dumps(self.server.server_config)

    def send_reset(self):
        msg = GLOBAL_MSG_TEMPLATE
        msg["payload"]["reset_episode"] = True
        self.server.msg = json.dumps(msg)

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def write_image_to_file_incrementally(self, image):
        """
        Dumping the image to a continuously progressing file, just for debugging purposes
        """
        os.makedirs(self.server.debug_config["image_dir"], exist_ok=True)

        i = 1
        while os.path.exists(os.path.join(self.server.debug_config["image_dir"], f"sample{i}.jpeg")):
            i += 1
        with open(os.path.join(self.server.debug_config["image_dir"], f"sample{i}.jpeg"), "wb") as f:
            f.write(image)

    def is_game_over(self):
        return self.over






    



    
    