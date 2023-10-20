import base64
import logging
import os
import shutil
import time
import numpy as np
import math
from io import BytesIO
from PIL import Image
from skimage.transform import resize
from python_server import PythonServer

logger = logging.getLogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#
def write_image_to_file_incrementally(image, image_dir):
    """
    Dumping the image to a continuously progressing file, just for debugging purposes. Keep most recent 1,000 images only.
    """
    with open(image_dir, 'wb') as f:
        f.write(image)


def clean_and_remake(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


class UnitySimHandler:
    def __init__(self, opt_d, max_d, img_scale, debug, protocol, host):
        self.debug_logs = debug
        self.debug_logs_dir = os.path.abspath(os.path.join(os.pardir, 'Logs'))
        self.image_dir = os.path.abspath(os.path.join(self.debug_logs_dir, 'images'))
        self.packets_sent_dir = os.path.abspath(os.path.join(self.debug_logs_dir, 'packets_sent'))
        self.packets_received_dir = os.path.abspath(os.path.join(self.debug_logs_dir, 'packets_received'))
        self.sim_ready = False
        self.server_connected = False
        self.img_scale = img_scale
        self.image_array = np.zeros(img_scale)
        self.last_obs = np.zeros(img_scale)
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.a = 0.0
        self.opt_d = opt_d
        self.max_d = max_d

        self.fns = {
            "connection_request": self.connection_request,
            'client_ready': self.sim_ready_request,
            "on_telemetry": self.on_telemetry
        }

        logger.setLevel(logging.INFO)
        self.server = PythonServer(self, protocol, host)

    def wait_until_loaded(self):
        while not self.server_connected:
            logger.warning("waiting for sim...")
            time.sleep(1.0)

    def wait_until_client_ready(self):
        while not self.sim_ready:
            logger.warning("waiting for client...")
            time.sleep(1.0)

    def render(self):
        pass

    def quit(self):
        self.server.stop()

    def clean_and_create_debug_directories(self):
        clean_and_remake(self.image_dir)
        clean_and_remake(self.packets_sent_dir)
        clean_and_remake(self.packets_received_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Gym ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def reset(self):
        logger.debug("resetting")
        self.server.action_num = 0
        self.server.episode_num += 1
        self.image_array = np.zeros((256, 256, 3))
        self.last_obs = self.image_array
        self.hit = []
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.a = 0.0
        self.send_reset()

    def observe(self, obs):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array

        # for vector obs training run, overwrite image observation with vector obs 
        # observation and self.last_obs left in because orchestrates above while loop which is making Python server wait for next message from client
        if obs == 'vector':
            observation = [self.rover_pos[0], self.rover_pos[1], self.rover_pos[2], self.rover_fwd[0],
                           self.rover_fwd[1], self.rover_fwd[2],
                           self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_fwd[0],
                           self.target_fwd[1], self.target_fwd[2]]

        info = {"rov_pos": self.rover_pos, "targ_pos": self.target_pos, "dist": self.raw_d, "raw_dist": self.raw_d,
                "rov_fwd": self.rover_fwd, "targ_fwd": self.target_fwd, "ang_error": self.a}

        return observation, self.calc_reward(), self.determine_episode_over(), False, info

    def calc_reward(self):
        # heading vector from rover to target
        heading = self.target_pos - self.rover_pos

        # normalize
        norm_heading = heading / np.linalg.norm(heading)

        # calculate radial distance on the flat y-plane
        self.raw_d = math.sqrt(math.pow(heading[0], 2) + math.pow(heading[2], 2))

        # calculate angle between rover's forward facing vector and heading vector
        self.a = math.degrees(
            math.atan2(norm_heading[0], norm_heading[2]) - math.atan2(self.rover_fwd[0], self.rover_fwd[2]))

        # scaling function producing value in the range [-1, 1] - distance and angle equal contribution
        reward = 1.0 - ((math.pow((self.raw_d - self.opt_d), 2) / math.pow(self.max_d, 2)) + (math.fabs(self.a) / 180))

        return reward

    def determine_episode_over(self):
        if math.fabs(self.raw_d - self.opt_d) > self.max_d:
            print("Episode terminated as target out of range: {}".format(abs(self.raw_d - self.opt_d)))
            logger.debug(f"game over: distance {self.raw_d}")
            return True
        if "Dolphin" in self.hit:
            print("Episode terminated due to collision")
            logger.debug(f"game over: hit {self.hit}")
            return True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Socket ~~~~~~~~~~~~~~~~~~~~~~~~~#

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
        logger.debug("socket connected")
        self.server_connected = True
        return

    def sim_ready_request(self, payload):
        logger.debug('client ready')
        self.sim_ready = True

    def on_telemetry(self, payload):
        self.rover_pos = np.array([payload['telemetry_data']['position'][0], payload['telemetry_data']['position'][1],
                                   payload['telemetry_data']['position'][2]])
        self.hit = payload['telemetry_data']['collision_objects']
        self.rover_fwd = np.array([payload['telemetry_data']["fwd"][0], payload['telemetry_data']['fwd'][1],
                                   payload['telemetry_data']['fwd'][2]])

        # TODO: implement receiving json on multiple targets
        # Sam.A, targets are now an array, use the last element of it for targeting
        for target in payload['telemetry_data']['targets']:
            self.target_pos = np.array([target['position'][0], target['position'][1], target['position'][2]])
            self.target_fwd = np.array([target['fwd'][0], target['fwd'][1], target['fwd'][2]])

        image = bytearray(base64.b64decode(payload['telemetry_data']['jpg_image']))

        if self.debug_logs:
            write_image_to_file_incrementally(image, os.path.abspath(os.path.join(self.image_dir, 'episode_' + str(self.server.episode_num) + '_' + 'image{}.jpg'.format(payload['obsv_num']))))

        image = np.array(Image.open(BytesIO(image)))

        if image.shape != self.img_scale:
            image = resize(image, self.img_scale, 0).astype(np.uint8)

        self.image_array = image

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing comms ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def send_action(self, action):
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
                    'manualMode': '0',
                    'stabilizeMode': '0',
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

    def send_object_pos(self, object, pos_x, pos_y, pos_z, Q):
        # note that the pos_y argument has been allocated to the z-axis array index
        # and the pos_z argument has been allocated to the y-axis array index
        # due to the difference in axis naming between scipy (incoming) and unity (outgoing)
        self.server.msg = {
            'msgType': 'set_position',
            'payload': {
                'objectPositions': [
                    {
                        'object_name': object,
                        'position': [pos_x, pos_z, pos_y],
                        'rotation': Q
                    }
                ]
            }
        }