import logging
import time
import json
import numpy as np 
#from io import BytesIO
#from PIL import Image

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

        #self.image_array = np.zeros((256,256,3))
        self.image_array = None # just using None for now whilst not unpacking data
        self.last_obs = None

        #self.over = False

    #     self.fns = {
    #         "telemetry": self.on_telemetry,
    #     }

    # # ------------ Gym ------------ #

    # def reset(self):

    #     logger.debug("resetting")

    #     self.image_array = np.zeros(self.conf['input_dim'])
    #     self.last_obs = self.image_array

    #     self.over = False

    def take_action(self, action):
        if self.server is None:
            return
        self.server.msg = json.dumps(action)
    
    def observe(self):

        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array                                             
        #done = self.is_game_over()
        #reward = self.calc_reward(done)
                                              
        #info = {"dummy": "can add variables here"}                            

        #return observation, reward, done, info
        return observation

    # # ------------ RL ------------ #

    # def calc_reward(self, done):

    #     ####### can implement reward function once wrapper working #######

    #     reward = 1

    #     return reward

    # #def determine_episode_over(self):

    #     ####### implement episode termination criteria here ###########

    # # just an extra getter function
    # def is_game_over(self):
    #     return self.over

    # # ------------ Socket ------------ #

    def on_connect(self, server):
        logger.debug("socket connected")
        self.server = server

    def on_disconnect(self):
        logger.debug("socket disconnected")
        self.server = None

    def on_recv_message(self, message):
        #if "msg_type" not in message:
            #logger.warn("expected msg_type field")
            #return
        #msg_type = message["msg_type"]
        #logger.debug("got message :" + msg_type)
        #if msg_type in self.fns:
            #self.fns[msg_type](message)
        #else:
            #logger.warning(f"unknown message type {msg_type}")

        ##### when improve message format to be dict with msg_type as key and message as value, can use above code ######
        ##### for now, force telemetry call ########
        self.on_telemetry(message)

    def on_telemetry(self, data):

        ###### for now just assuming that data is image and nothing else #######
        ###### also, there will need to unpack data 
        ##### but for now just setting self.image_array as raw data #####

        #image = Image.open(BytesIO(data))
        #self.image_array = np.array(image)
        self.image_array = data     

        #if self.over:
            #return

        #self.determine_episode_over()

    def get_cam_config(self):
        return {
            "fov" : 100
        }

    def get_rover_config(self):
        return {
            "thrustPower" : 12
        }

    def get_server_config(self):
        jsonObj = {
            "serverConfig" : {
                "camConfig" : self.get_cam_config(),
                "roverConfig" : self.get_rover_config()
            }
        }
        return json.dumps(jsonObj)




    



    
    