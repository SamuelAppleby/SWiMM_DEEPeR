import base64
import json
import logging
import os
import shutil
import time
from enum import IntEnum
import socket
from threading import Thread

import cv2
import numpy as np
import math
from io import BytesIO
from PIL import Image
from jsonschema.validators import validate
from datetime import datetime


class Protocol(IntEnum):
    UDP = 0
    TCP = 1

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class EpisodeTerminationType(IntEnum):
    MAXIMUM_DISTANCE = 0
    TARGET_OUT_OF_VIEW = 1
    TARGET_COLLISION = 2
    THRESHOLD_REACHED = 3


def process_and_validate_configs(dir_map):
    arr = []
    for conf_dir, schema_dir in dir_map.items():
        assert os.path.isfile(conf_dir)
        assert os.path.isfile(schema_dir)

        with open(conf_dir) as file_conf:
            conf_json = json.load(file_conf)

            assert os.path.isfile(schema_dir)
            with open(schema_dir) as file_schema:
                schema_json = json.load(file_schema)
                validate(instance=conf_json, schema=schema_json)
                arr.append(conf_json)

    return arr


def clean_and_remake(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


logger = logging.getLogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#
def write_image_to_file_incrementally(image, directory):
    """
    Dumping the image to a continuously progressing file, just for debugging purposes. Keep most recent 1,000 images only.
    """
    with open(directory, 'wb') as f:
        f.write(image)


def on_disconnect():
    logger.debug("socket disconnected")


class UnitySimHandler:
    def __init__(self, opt_d, max_d, img_res, debug, protocol, host, ep_len_threshold, seed):
        self.sim_ready = False
        self.server_connected = False
        self.img_res = img_res
        self.image_array = np.zeros(self.img_res)
        self.image_array = np.zeros(self.img_res)
        self.last_obs = np.zeros(self.img_res)
        self.hit = False
        self.target_out_of_view = False
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.a = 0.0
        self.opt_d = opt_d
        self.max_d = max_d
        self.ep_length_threshold = ep_len_threshold
        self.seed = seed

        self.fns = {
            "connection_request": self.connection_request,
            'client_ready': self.sim_ready_request,
            "on_telemetry": self.on_telemetry
        }

        logger.setLevel(logging.INFO)

        self.debug_logs = debug
        self.debug_logs_dir = os.path.join(os.pardir, 'Logs')
        self.image_dir = os.path.join(self.debug_logs_dir, 'images')
        self.packets_sent_dir = os.path.join(self.debug_logs_dir, 'packets_sent')
        self.packets_received_dir = os.path.join(self.debug_logs_dir, 'packets_received')

        self.sock = None
        self.addr = None
        self.do_process_msgs = False
        self.msg = {}
        self.th = None
        self.conn = None
        self.network_config = None
        self.server_config = None
        self.action_buffer_size = None
        self.observation_buffer_size = None
        self.episode_num = -1
        self.step_num = 0
        self.episode_termination_type = EpisodeTerminationType.THRESHOLD_REACHED

        conf_arr = process_and_validate_configs({
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'server_config.json'): os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs',
                                                                                                                               'server_config_schema.json')
        })

        self.server_config = conf_arr.pop()
        self.server_config['payload']['serverConfig']['envConfig']['optD'] = self.opt_d
        self.server_config['payload']['serverConfig']['envConfig']['maxD'] = self.max_d
        self.server_config['payload']['serverConfig']['envConfig']['seed'] = self.seed

        self.protocol = protocol
        self.full_host = host.split(':')
        self.address = (self.full_host[0], (int(self.full_host[1])))
        self.observation_buffer_size = 8192

        if self.debug_logs:
            self.clean_and_create_debug_directories()

        self.connect(*self.address)

    def clean_and_create_debug_directories(self):
        clean_and_remake(self.image_dir)
        clean_and_remake(self.packets_sent_dir)
        clean_and_remake(self.packets_received_dir)

    def connect(self, host='127.0.0.1', port=60260):
        """
        Open a tcp/udp socket
        """

        # create socket and associate the socket with local address
        if self.protocol == Protocol.UDP:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock.bind((host, port))
        print(f"[+] Listening on {host}:{port}")

        # the remaining network related code, receiving json and sending json, is run in a thread
        self.do_process_msgs = True
        self.th = Thread(target=self.proc_msg, daemon=True)
        self.th.start()

    def proc_msg(self):
        """
        Continue to process messages from each connected client
        """

        if self.protocol == Protocol.TCP:
            try:
                # wait for connection request
                self.sock.listen(5)
                self.conn, self.addr = self.sock.accept()
                current_time = datetime.now().ctime()
                print(
                    f"[+] Connecting by {self.addr[0]}:{self.addr[1]} ({current_time})")
            except ConnectionRefusedError as refuse_error:
                raise (Exception("Could not connect to server.")) from refuse_error

        while self.do_process_msgs:
            # print('Waiting to receive message')
            # receive packets/datagrams
            if self.protocol == Protocol.UDP:
                data, self.addr = self.sock.recvfrom(self.observation_buffer_size)
            else:
                data = self.conn.recv(self.observation_buffer_size)

            if not data:
                print("[-] Invalid json")
                self.stop()
                return

            # unpack and send json message
            json_str = data.decode('UTF-8')
            # print('Received: {}'.format(json_str))
            json_dict = json.loads(json_str)

            if self.debug_logs:
                with open(os.path.join(self.packets_received_dir, 'episode_{}_step_{}.json'.format(self.episode_num, self.step_num)), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f, ensure_ascii=False, indent=4)
                    f.close()

            self.on_recv_message(json_dict)

            # wait to point something to self.msg variable dedicated to outgoing messages
            # print('Waiting to send message')
            while self.msg == {}:
                time.sleep(1.0 / 120.0)

            # print('Sending: {}'.format(str(self.msg.encode('utf-8'))))
            json_str = json.dumps(self.msg)
            # print('Sending: {}'.format(json_str))

            if self.debug_logs:
                if self.msg['msgType'] == "reset_episode":
                    self.clean_and_create_debug_directories()
                else:
                    with open(os.path.join(self.packets_sent_dir, 'episode_{}_step_{}.json'.format(self.episode_num, self.step_num)), 'w', encoding='utf-8') as f:
                        json.dump(self.msg, f, ensure_ascii=False, indent=4)
                        f.close()

            if self.protocol == Protocol.UDP:
                self.sock.sendto(bytes(json_str, encoding="utf-8"), self.addr)
            else:
                self.conn.sendall(bytes(json_str, encoding="utf-8"))

            self.msg = {}

    def stop(self):
        """
        Signal proc_msg loop to stop, wait for thread to finish, close socket
        """
        self.do_process_msgs = False

        self.msg = {
            'msgType': 'endSimulation',
            'payload': {
            }
        }

        if self.sock is not None:
            print("[-] Closing socket")
            self.sock.close()
            on_disconnect()

    def wait_until_loaded(self):
        while not self.server_connected:
            logger.info("waiting for sim...")
            time.sleep(1.0)

    def wait_until_client_ready(self):
        while not self.sim_ready:
            logger.info("waiting for client...")
            time.sleep(1.0)

    def render(self):
        pass

    def quit(self):
        self.stop()

    def reset(self, ep_n):
        logger.debug("resetting")
        self.image_array[:] = 0
        self.last_obs = self.image_array
        self.hit = False
        self.target_out_of_view = False
        self.rover_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.rover_fwd = np.zeros(3)
        self.target_fwd = np.zeros(3)
        self.raw_d = 0.0
        self.a = 0.0
        self.episode_num = ep_n
        self.step_num = 0
        self.msg = {
            'msgType': 'resetEpisode',
            'payload': {}
        }

    def observe(self, obs):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array

        # for vector obs training run, overwrite image observation with vector obs 
        # observation and self.last_obs left in because orchestrates above while loop which is making Python server wait for next message from client
        if obs == 'vector':
            observation = [self.rover_pos[0], self.rover_pos[1], self.rover_pos[2],
                           self.rover_fwd[0], self.rover_fwd[1], self.rover_fwd[2],
                           self.target_pos[0], self.target_pos[1], self.target_pos[2],
                           self.target_fwd[0], self.target_fwd[1], self.target_fwd[2]]

        info = {"rov_pos": self.rover_pos, "targ_pos": self.target_pos, "dist": self.raw_d, "raw_dist": self.raw_d,
                "rov_fwd": self.rover_fwd, "targ_fwd": self.target_fwd, "ang_error": self.a}

        return observation, self.calc_reward(), self.determine_episode_over(), False, info

    def calc_reward(self):
        # heading vector from rover to target
        heading = self.target_pos - self.rover_pos

        # normalize
        norm_heading = heading / np.linalg.norm(heading)

        # calculate angle between rover's forward facing vector and heading vector
        self.a = math.degrees(
            math.atan2(norm_heading[0], norm_heading[2]) - math.atan2(self.rover_fwd[0], self.rover_fwd[2]))

        # calculate radial distance on the flat y-plane
        self.raw_d = math.sqrt(math.pow(heading[0], 2) + math.pow(heading[2], 2))

        # scaling function producing value in the range [-1, 1] - distance and angle equal contribution
        reward = 1.0 - ((math.pow((self.raw_d - self.opt_d), 2) / math.pow(self.max_d, 2)) + (math.fabs(self.a) / 180))

        return reward

    def determine_episode_over(self):
        if self.hit:
            print('[EPISODE TERMINATED] Due to collision with target')
            logger.debug(f"game over: target hit")
            self.episode_termination_type = EpisodeTerminationType.TARGET_COLLISION
            return True
        if self.target_out_of_view:
            print('[EPISODE TERMINATED] Due to target out of view')
            logger.debug(f"game over: target out of view")
            self.episode_termination_type = EpisodeTerminationType.TARGET_OUT_OF_VIEW
            return True
        if (self.ep_length_threshold > 0) and (self.step_num == self.ep_length_threshold):
            print('[EPISODE TERMINATED]: Maximum episode length reached: {}'.format(self.ep_length_threshold))
            logger.debug(f"game over: episode threshold {self.ep_length_threshold}")
            self.episode_termination_type = EpisodeTerminationType.THRESHOLD_REACHED
            return True
        if math.fabs(self.raw_d - self.opt_d) > self.max_d:
            print('[EPISODE TERMINATED] Too far from the optimum distance: {}'.format(abs(self.raw_d - self.opt_d)))
            logger.debug(f"game over: distance {self.raw_d}")
            self.episode_termination_type = EpisodeTerminationType.MAXIMUM_DISTANCE
            return True
        return False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def on_recv_message(self, message):
        if "msgType" not in message:
            logger.warning("expected msgType field")
            return
        msg_type = message["msgType"]
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
        self.rover_pos = np.array([payload['telemetryData']['position'][0], payload['telemetryData']['position'][1],
                                   payload['telemetryData']['position'][2]])
        self.rover_fwd = np.array([payload['telemetryData']["fwd"][0], payload['telemetryData']['fwd'][1],
                                   payload['telemetryData']['fwd'][2]])

        for target in payload['telemetryData']['targets']:
            self.target_pos = np.array([target['position'][0], target['position'][1], target['position'][2]])
            self.target_fwd = np.array([target['fwd'][0], target['fwd'][1], target['fwd'][2]])
            self.hit = target['colliding']
            self.target_out_of_view = target['outOfView']

        image = bytearray(base64.b64decode(payload['telemetryData']['jpgImage']))

        if self.debug_logs:
            write_image_to_file_incrementally(image, os.path.join(self.image_dir, 'episode_{}_step{}.jpg'.format(self.episode_num, self.step_num)))

        image = np.array(Image.open(BytesIO(image)))

        if image.shape != self.img_res:
            image = cv2.resize(image, self.img_res).astype(np.uint8)

        self.image_array = image

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def send_action(self, action, step_n):
        self.msg = {
            'msgType': 'receiveJsonControls',
            'payload': {
                'jsonControls': {
                    'swayThrust': '0',
                    'heaveThrust': '0',
                    'surgeThrust': action[0].__str__(),
                    'pitchThrust': '0',
                    'yawThrust': action[1].__str__(),
                    'rollThrust': '0',
                    'mode': '0',
                }
            }
        }

        self.step_num = step_n

    def send_server_config(self):
        """
        Generate server config for client
        """
        self.msg = self.server_config

    def send_object_pos(self, object_pos, pos_x, pos_y, pos_z, q):
        # note that the pos_y argument has been allocated to the z-axis array index
        # and the pos_z argument has been allocated to the y-axis array index
        # due to the difference in axis naming between scipy (incoming) and unity (outgoing)
        self.msg = {
            'msgType': 'setPosition',
            'payload': {
                'objectPositions': [
                    {
                        'objectName': object_pos,
                        'position': [pos_x, pos_z, pos_y],
                        'rotation': q
                    }
                ]
            }
        }
