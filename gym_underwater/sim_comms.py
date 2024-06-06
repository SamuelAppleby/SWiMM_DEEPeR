import base64
import json
import os
import shutil
import sys
import time
import socket
import threading

import numpy as np
import math
from jsonschema.validators import validate
from datetime import datetime

from stable_baselines3.common.type_aliases import TrainFrequencyUnit

import cmvae_utils.dataset_utils
from gym_underwater.constants import PORT_TRAIN
from gym_underwater.enums import Protocol, EpisodeTerminationType, TrainingType


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#
def calc_metrics(rov_pos, rov_fwd, target_pos):
    # heading vector from rover to target
    heading = target_pos - rov_pos

    # normalize
    norm_heading = heading / np.linalg.norm(heading)

    # calculate radial distance on the flat y-plane
    raw_d = math.sqrt(math.pow(heading[0], 2) + math.pow(heading[2], 2))

    # calculate angle between rover's forward facing vector and heading vector
    dot = np.dot(norm_heading, rov_fwd)

    # floating-point inaccuracy may cause epsilon violations, so clamp to legal values
    dot = np.clip(dot, -1, 1)
    acos = np.arccos(dot)

    a = np.degrees(acos)

    assert not math.isnan(a)

    return raw_d, a


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


MAX_STEP_REWARD = 1.0
PERIOD = 120


class ClientDisconnectedException(Exception):
    pass


class UnitySimHandler:
    def __init__(self, opt_d, max_d, img_res, tensorboard_log, protocol, ip, port, seed, cancel_event=None, read_write_thread_other=None):
        self.interval = 1 / PERIOD
        self.timeout = self.interval * PERIOD * 60 * 10  # Timeout will occur if 10 minutes have occurred without message
        self.sim_ready = False
        self.last_obs = None
        self.rover_info = None
        self.target_info = None
        self.opt_d = opt_d
        self.max_d = max_d
        self.msg_event = threading.Event()
        self.image_array = None
        self.img_event = threading.Event()
        self.cancel_event = threading.Event() if cancel_event is None else cancel_event
        self.msg = None
        self.debug_logs = False

        self.fns = {
            'clientReady': self.on_client_ready,
            'onTelemetry': self.on_telemetry
        }

        self.episode_num = 0
        self.packet_received_num = 0
        self.packet_sent_num = 0
        self.image_num = 0

        self.read_write_thread_other = read_write_thread_other

        conf_arr = process_and_validate_configs({
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'server_config.json'): os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'schemas',
                                                                                                                               'server_config_schema.json')
        })

        self.server_config = conf_arr.pop()
        self.server_config['payload']['serverConfig']['envConfig']['optD'] = self.opt_d
        self.server_config['payload']['serverConfig']['envConfig']['maxD'] = self.max_d
        self.server_config['payload']['serverConfig']['envConfig']['seed'] = seed

        self.training_type = TrainingType.TRAINING if port == PORT_TRAIN else TrainingType.INFERENCE

        if self.debug_logs is not None:
            self.debug_logs_dir = os.path.join(tensorboard_log, 'network', 'training' if self.training_type == TrainingType.TRAINING else os.path.join(tensorboard_log, 'network', 'inference'),
                                               f'episode_{self.episode_num}')
            clean_and_remake(self.debug_logs_dir)
            self.clean_and_create_debug_directories()

        self.sock = None
        self.addr = None
        self.conn = None
        self.action_buffer_size = None
        self.threads = []

        self.protocol = protocol
        self.address = (ip, port)

        self.observation_buffer_size = 8192
        self.img_res = img_res

        self.read_write_thread = self.connect(*self.address)
        self.read_write_thread.start()

    def connect(self, host='127.0.0.1', port=60260) -> threading.Thread:
        """
        Open a tcp/udp socket
        """
        # create socket and associate the socket with local address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM if self.protocol == Protocol.UDP else socket.SOCK_STREAM)

        self.sock.bind((host, port))
        print(f'Listening on {host}:{port}')

        if self.protocol == Protocol.TCP:
            while self.conn is None:
                try:
                    self.sock.listen()
                    self.conn, self.addr = self.sock.accept()
                    self.conn.settimeout(self.timeout)
                except ConnectionRefusedError as refuse_error:
                    print('[NETWORK ERROR] Connection refused:', refuse_error)
        else:
            self.sock.settimeout(self.timeout)

        print(f'Connecting by {self.addr[0]}:{self.addr[1]} ({datetime.now().ctime()})')
        return threading.Thread(target=self.continue_read_write, daemon=True)

    def set_obsv(self, image):
        while self.img_event.is_set():
            time.sleep(self.interval)
        self.image_array = image
        self.img_event.set()

    def set_msg(self, msg):
        # For some rapidly occurring messages, we may have to briefly wait until the event is free
        while self.msg_event.is_set():
            time.sleep(self.interval)

        self.msg = msg
        self.msg_event.set()

    def stop(self):
        """
        Signal proc_msg loop to stop, wait for thread to finish, close socket
        """
        self.set_msg({
            'msgType': 'endSimulation',
            'payload': {
            }
        })

    def render(self):
        pass

    def reset(self):
        self.image_array = None
        self.last_obs = None
        self.rover_info = None
        self.target_info = None
        self.set_msg({
            'msgType': 'resetEpisode',
            'payload': {}
        })

    def observe(self, obs):
        while not self.img_event.is_set() and self.read_write_thread.is_alive():
            time.sleep(self.interval)

        if self.cancel_event.is_set():
            self.read_write_thread.join()

            # Our evaluation/training environments needs to consider each-other
            if self.read_write_thread_other is not None:
                self.read_write_thread_other.join()

            sys.exit(0)

        if self.last_obs is not None:
            assert not np.array_equal(self.image_array, self.last_obs)

        self.last_obs = self.image_array
        observation = self.image_array

        info = {
            'rover': self.rover_info,
            'target': self.target_info
        }

        # for vector obs training run, overwrite image observation with vector obs
        # observation and self.last_obs left in because orchestrates above while loop which is making Python server wait for next message from client
        if obs == 'vector':
            observation = np.array([info['rover']['pos'][0], info['rover']['pos'][1], info['rover']['pos'][2], info['rover']['fwd'][0], info['rover']['fwd'][1], info['rover']['fwd'][2],
                                    info['target']['pos'][0], info['target']['pos'][1], info['target']['pos'][2], info['target']['fwd'][0], info['target']['fwd'][1], info['target']['fwd'][2]])

        raw_d, a = calc_metrics(np.array(info['rover']['pos']), np.array(info['rover']['fwd']), np.array(info['target']['pos']))
        reward = self.calc_reward(raw_d, a)

        over = None

        # During inference, we still allow TimeLimit truncation but want to simulate the real-world inference, so no early reset
        if self.training_type == TrainingType.TRAINING:
            over = self.determine_episode_over(raw_d)

        info.update({
            'dist': raw_d,
            'ang_error': a,
            'episode_termination_type': over
        })

        # During training, guarantee that reward is between -1 and 1 if not over
        if (self.training_type == TrainingType.TRAINING) and over is None:
            assert (-1 <= reward <= 1)

        self.img_event.clear()

        return observation, reward, True if over is not None else False, False, info

    def calc_reward(self, raw_d, a):
        # scaling function producing value in the range [-1, 1] - distance and angle equal contribution
        return (MAX_STEP_REWARD - ((math.pow((raw_d - self.opt_d), 2) / math.pow(self.max_d, 2)) + (math.fabs(a) / 180)))

    def determine_episode_over(self, raw_d):
        if self.target_info['colliding']:
            print('[EPISODE TERMINATED] Due to collision with target')
            return EpisodeTerminationType.TARGET_COLLISION
        if self.target_info['outOfView']:
            print('[EPISODE TERMINATED] Due to target out of view')
            return EpisodeTerminationType.TARGET_OUT_OF_VIEW
        if abs(raw_d - self.opt_d) > self.max_d:
            print(f'[EPISODE TERMINATED] Too far from the optimum distance: {abs(raw_d - self.opt_d)}')
            return EpisodeTerminationType.MAXIMUM_DISTANCE
        return None

    def continue_read_write(self):
        self.threads = [
            threading.Thread(target=self.recv_msg, daemon=True),
            threading.Thread(target=self.send_msg, daemon=True)
        ]

        for t in self.threads:
            t.start()

        for t in self.threads:
            t.join()

        return

    def disconnect(self):
        self.addr = None

        if self.conn is not None:
            self.conn.close()
            self.conn = None
        elif self.sock is not None:
            self.sock.close()

        self.sock = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#
    def recv_msg(self):
        """
        Continue to process messages from each connected client
        """
        while not self.cancel_event.is_set():
            try:
                if self.protocol == Protocol.UDP:
                    data, self.addr = self.sock.recvfrom(self.observation_buffer_size)
                else:
                    data = self.conn.recv(self.observation_buffer_size)
                if not data:
                    # This is the normal flow, i.e. when the client closes the connection
                    raise ClientDisconnectedException(f'Client closed the connection')

                json_str = data.decode('UTF-8')
                # print(f'Received: {json_str}')
                json_dict = json.loads(json_str)

                if self.debug_logs_dir is not None:
                    with open(os.path.join(self.debug_logs_dir, 'packets_received', f'step_{self.packet_received_num}.json'), 'w', encoding='utf-8') as f:
                        json.dump(json_dict, f, indent=2)
                        f.close()
                        self.packet_received_num += 1

                self.on_recv_message(json_dict)

            except Exception as e:
                print('[NETWORK ERROR] Read error:', e)
                self.cancel_event.set()

    def on_recv_message(self, message):
        assert 'msgType' in message, 'msgType not found in message'

        msg_type = message['msgType']
        payload = message['payload']

        assert msg_type in self.fns, f'Unknown message type {msg_type}'
        self.fns[msg_type](payload)

    def on_client_ready(self, payload):
        self.sim_ready = True

    def on_telemetry(self, payload):
        self.rover_info = payload['telemetryData']['rover']
        self.target_info = payload['telemetryData']['target']

        image = bytearray(base64.b64decode(self.rover_info['obs']))

        if self.debug_logs_dir is not None:
            with open(os.path.join(self.debug_logs_dir, 'images', f'step_{self.image_num}.jpg'), 'wb') as f:
                f.write(image)

            self.image_num += 1

        image = cmvae_utils.dataset_utils.load_img_from_file_or_array_and_resize_cv2(img_array=image, res=self.img_res, normalise=True)

        self.set_obsv(image)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#
    def send_msg(self):
        """
        Continue to process messages from each connected client
        """
        while not self.cancel_event.is_set():
            try:
                while not self.msg_event.is_set() and not self.cancel_event.is_set():
                    time.sleep(self.interval)

                if self.cancel_event.is_set():
                    continue

                assert self.msg is not None

                json_str = json.dumps(self.msg)
                # print(f'Sending: {json_str}')

                if self.debug_logs_dir is not None:
                    with open(os.path.join(self.debug_logs_dir, 'packets_sent', f'step_{self.packet_sent_num}.json'), 'w', encoding='utf-8') as f:
                        json.dump(self.msg, f, indent=2)
                        f.close()
                        self.packet_sent_num += 1

                    match self.msg['msgType']:
                        case 'resetEpisode':
                            self.episode_num += 1
                            self.debug_logs_dir = self.debug_logs_dir[:self.debug_logs_dir.rfind('_') + 1] + str(self.episode_num)
                            self.clean_and_create_debug_directories()
                            self.packet_received_num = 0
                            self.packet_sent_num = 0
                            self.image_num = 0
                        case _:
                            pass

                if self.protocol == Protocol.UDP:
                    self.sock.sendto(bytes(json_str, encoding='utf-8'), self.addr)
                else:
                    self.conn.sendall(bytes(json_str, encoding='utf-8'))

                self.msg = None
                self.msg_event.clear()
            except Exception as e:
                print('[NETWORK ERROR] Write error:', e)
                self.cancel_event.set()

        # Let's limit disconnection from here only
        self.disconnect()

    def send_action(self, action):
        self.set_msg({
            'msgType': 'action',
            'payload': {
                'jsonControls': {
                    'swayThrust': 0.0,
                    'heaveThrust': 0.0,
                    'surgeThrust': float(action[0]),
                    'pitchThrust': 0.0,
                    'yawThrust': float(action[1]),
                    'rollThrust': 0.0,
                    'mode': 0,
                }
            }
        })

    def send_server_config(self):
        """
        Generate server config for client
        """
        self.set_msg(self.server_config)

    def send_object_pos(self, object_pos, pos_x, pos_y, pos_z, q):
        # note that the pos_y argument has been allocated to the z-axis array index
        # and the pos_z argument has been allocated to the y-axis array index
        # due to the difference in axis naming between scipy (incoming) and unity (outgoing)
        self.set_msg({
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
        })

    def send_rollout_start(self):
        self.set_msg({
            'msgType': 'rolloutStart',
            'payload': {}
        })

    def send_rollout_end(self):
        self.set_msg({
            'msgType': 'rolloutEnd',
            'payload': {}
        })

    def send_inference_start(self):
        self.set_msg({
            'msgType': 'inferenceStart',
            'payload': {
            }
        })

    def send_world_state(self, world_state):
        self.set_msg({
            'msgType': 'setWorldState',
            'payload': {
                'objectOverrides': world_state
            }
        })

    def send_inference_end(self):
        self.set_msg({
            'msgType': 'inferenceEnd',
            'payload': {}
        })

    def clean_and_create_debug_directories(self):
        clean_and_remake(self.debug_logs_dir)
        clean_and_remake(os.path.join(self.debug_logs_dir, 'packets_received'))
        clean_and_remake(os.path.join(self.debug_logs_dir, 'packets_sent'))
        clean_and_remake(os.path.join(self.debug_logs_dir, 'images'))

    def send_end_simulation(self):
        self.set_msg({
            'msgType': 'endSimulation',
            'payload': {}
        })

    def close(self):
        self.send_end_simulation()
        self.read_write_thread.join()
