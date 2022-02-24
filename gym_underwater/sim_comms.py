import logging
import time
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

    # #def wait_until_loaded(self):
    #     #while not self.handler.loaded:
    #         #logger.warning("waiting for sim to start..")
    #         #time.sleep(3.0)          

    # def reset(self):
    #     self.handler.reset()

    # def take_action(self, action):
    #     self.handler.take_action(action)

    # def observe(self):
    #     return self.handler.observe()

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
        #self.loaded = False

        self.image_array = np.zeros((256,256,3))
        self.last_obs = None

        #self.over = False

    #     self.fns = {
    #         "telemetry": self.on_telemetry,
    #         #"sim_loaded": self.on_sim_loaded,
    #     }

    def on_connect(self, server):
        logger.debug("socket connected")
        self.server = server

    # #def on_disconnect(self):
    #     #logger.debug("socket disconnected")
    #     #self.server = None

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

    # # ------------ Gym ------------ #

    # def reset(self):

    #     logger.debug("resetting")

    #     self.image_array = np.zeros(self.conf['input_dim'])
    #     self.last_obs = self.image_array

    #     self.over = False

    # def take_action(self, action):
    #     self.send_control_message(action)

    # def observe(self):

    #     while self.last_obs is self.image_array:
    #         time.sleep(1.0 / 120.0)

    #     self.last_obs = self.image_array
    #     observation = self.image_array                                             
    #     done = self.is_game_over()
    #     reward = self.calc_reward(done)
                                              
    #     info = {"dummy": "can add variables here"}                            

    #     return observation, reward, done, info

    # def is_game_over(self):
    #     return self.over

    # # ------------ RL ------------ #

    # def calc_reward(self, done):

    #     ####### can implement reward function once wrapper working #######

    #     reward = 1

    #     return reward

    # # ------------ Socket ------------ #

    def on_telemetry(self, data):

        ###### for now just assuming that data is image and nothing else #######

        binary_image = data
        print("Received by handler")
        #image = Image.open(BytesIO(binary_image))
        #self.image_array = np.array(image)
        #if self.over:
            #return

        #self.determine_episode_over()

    # #def determine_episode_over(self):

    #     ####### implement episode termination criteria here ###########

    # #def on_sim_loaded(self):
    #     #logger.debug("sim loaded")
    #     #self.loaded = True
    
    # def send_control_message(self, action):
    #     #if not self.loaded:
    #         #return
    #     msg = {
    #         "forwardThrust": action[0],
    #         "verticalThrust": action[1],
    #         "yRotation": action[2],
    #     }
    #     self.queue_message(msg)

    # def queue_message(self, msg):
    #     if self.server is None:
    #         logger.debug(f"skiping: \n {msg}")
    #         return

    #     logger.debug(f"sending \n {msg}")
    #     self.server.queue_message(msg)
    
    