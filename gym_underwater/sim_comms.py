import base64
import json
import os
import shutil
import time
import socket
import threading
import cv2
import numpy as np
import math
from io import BytesIO
from PIL import Image
from jsonschema.validators import validate
from datetime import datetime

from stable_baselines3.common.type_aliases import TrainFrequencyUnit

from gym_underwater.enums import Protocol, EpisodeTerminationType, TrainingType


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#
def write_image_to_file_incrementally(image, directory):
    """
    Dumping the image to a continuously progressing file, just for debugging purposes. Keep most recent 1,000 images only.
    """
    with open(directory, 'wb') as f:
        f.write(image)


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


class UnitySimHandler:
    def __init__(self, opt_d, max_d, img_res, tensorboard_log, protocol, host, seed):
        self.sim_ready = False
        self.server_connected = False
        self.img_res = img_res
        self.last_obs = np.zeros(self.img_res)
        self.rover_info = None
        self.target_info = None
        self.opt_d = opt_d
        self.max_d = max_d
        self.seed = seed
        self.msg_event = threading.Event()
        self.image_array = np.zeros(self.img_res)
        self.img_event = threading.Event()
        self.cancel_event = threading.Event()
        self.training_type = TrainingType.TRAINING
        self.msg = None

        self.fns = {
            "connectionRequest": self.connection_request,
            'clientReady': self.sim_ready_request,
            "onTelemetry": self.on_telemetry
        }

        self.episode_num = -1
        self.inference_test_num = (-1, -1)
        self.eval_inference_freq = None
        self.eval_packet_sent = 0
        self.packet_received_num = 0
        self.packet_sent_num = 0
        self.image_num = 0

        self.tensorboard_log = os.path.join(tensorboard_log, 'network') if tensorboard_log is not None else None
        if self.tensorboard_log is not None:
            self.debug_logs_dir = os.path.join(self.tensorboard_log, 'training', f'episode_{self.episode_num}')
            clean_and_remake(os.path.dirname(os.path.dirname(self.debug_logs_dir)))
            self.clean_and_create_debug_directories()

        self.sock = None
        self.addr = None
        self.msg_discard = False
        self.conn = None
        self.network_config = None
        self.server_config = None
        self.action_buffer_size = None
        self.observation_buffer_size = None
        self.threads = []

        conf_arr = process_and_validate_configs({
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'server_config.json'): os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs',
                                                                                                                               'server_config_schema.json')
        })

        self.server_config = conf_arr.pop()
        self.server_config['payload']['serverConfig']['envConfig']['optD'] = self.opt_d
        self.server_config['payload']['serverConfig']['envConfig']['maxD'] = self.max_d
        self.server_config['payload']['serverConfig']['envConfig']['seed'] = self.seed

        self.protocol = protocol
        full_host = host.split(':')
        self.address = (full_host[0], (int(full_host[1])))

        self.observation_buffer_size = 8192
        self.read_write_thread = None

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
        print(f'Listening on {host}:{port}')

        if self.protocol == Protocol.TCP:
            try:
                # wait for connection request
                self.sock.listen(5)
                self.conn, self.addr = self.sock.accept()
                print(f'Connecting by {self.addr[0]}:{self.addr[1]} ({datetime.now().ctime()})')
                self.server_connected = True
            except ConnectionRefusedError as refuse_error:
                raise Exception('Could not connect to server.') from refuse_error

        self.read_write_thread = threading.Thread(target=self.continue_read_write, daemon=True)

    def set_obsv(self, image):
        while self.img_event.is_set():
            time.sleep(1 / 120)
        self.image_array = image
        self.img_event.set()

    def set_msg(self, msg):
        # For some rapidly occurring messages, we may have to briefly wait until the event is free
        while self.msg_event.is_set():
            time.sleep(1 / 120)

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
        self.image_array[:] = 0
        self.last_obs = self.image_array
        self.rover_info = None
        self.target_info = None
        self.set_msg({
            'msgType': 'resetEpisode',
            'payload': {}
        })

    def observe(self, obs):
        while not any(event.is_set() for event in [self.img_event, self.cancel_event]):
            time.sleep(1 / 120)

        if self.cancel_event.is_set():
            self.read_write_thread.join()
            # Our TCP socket might be bad, as training connect possibly recover, let's just raise the exception here
            raise Exception(f'The network socket: {self.address[0]}:{self.address[1]} is bad, quitting')

        assert self.image_array is not self.last_obs

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

        # During training,we guarantee that reward is between
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

        while all(t.is_alive() for t in self.threads):
            time.sleep(1 / 120)

        self.cancel_event.set()

        for t in self.threads:
            t.join()

        self.msg_event.clear()
        self.img_event.clear()

        if self.sock is not None:
            print('Closing socket')
            self.sock.close()
            print('Socket disconnected')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#
    def recv_msg(self):
        """
        Continue to process messages from each connected client
        """
        while not self.cancel_event.is_set():
            if self.protocol == Protocol.UDP:
                data, self.addr = self.sock.recvfrom(self.observation_buffer_size)
            else:
                data = self.conn.recv(self.observation_buffer_size)

            if not data:
                print('Invalid json')
                self.set_msg({
                    'msgType': 'internalQuit',
                    'payload': {
                    }
                })
                return

            json_str = data.decode('UTF-8')

            # print(f'Received: {json_str}')

            json_dict = json.loads(json_str)

            if (self.tensorboard_log is not None) and not self.msg_discard:
                with open(os.path.join(self.debug_logs_dir, 'packets_received', f'step_{self.packet_received_num}.json'), 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f, indent=2)
                    f.close()
                    self.packet_received_num += 1

            self.on_recv_message(json_dict)

            if self.msg_discard:
                self.msg_discard = False

    def on_recv_message(self, message):
        if 'msgType' not in message:
            return
        msg_type = message['msgType']
        payload = message['payload']
        if msg_type in self.fns:
            self.fns[msg_type](payload)
        else:
            print(f'unknown message type {msg_type}')

    def connection_request(self, payload):
        return

    def sim_ready_request(self, payload):
        self.sim_ready = True

    def on_telemetry(self, payload):
        self.rover_info = payload['telemetryData']['rover']
        self.target_info = payload['telemetryData']['target']

        image = bytearray(base64.b64decode(self.rover_info['obs']))

        if (self.tensorboard_log is not None) and not self.msg_discard:
            write_image_to_file_incrementally(image, os.path.join(self.debug_logs_dir, 'images', f'step_{self.image_num}.jpg'))
            self.image_num += 1

        image = np.array(Image.open(BytesIO(image)))

        if image.shape != self.img_res:
            image = cv2.resize(image, (self.img_res[0], self.img_res[1])).astype(np.uint8)

        self.set_obsv(image)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#
    def send_msg(self):
        """
        Continue to process messages from each connected client
        """
        while not self.cancel_event.is_set():
            while not self.msg_event.is_set():
                time.sleep(1 / 120)

            assert self.msg is not None

            json_str = json.dumps(self.msg)
            # print(f'Sending: {json_str}')

            if self.tensorboard_log is not None:
                match self.msg['msgType']:
                    case 'inferenceStart':
                        self.inference_test_num = (self.inference_test_num[0] + 1, -1)
                        self.packet_sent_num = 0
                        self.eval_packet_sent = 0
                        self.debug_logs_dir = os.path.join(self.tensorboard_log, 'inference', f'run_{self.inference_test_num[0]}', f'episode_{self.inference_test_num[1]}')
                        self.clean_and_create_debug_directories()
                    case 'action':
                        if self.training_type == TrainingType.INFERENCE:
                            self.eval_packet_sent += 1
                    case _:
                        pass

                with open(os.path.join(self.debug_logs_dir, 'packets_sent', f'step_{self.packet_sent_num}.json'), 'w', encoding='utf-8') as f:
                    json.dump(self.msg, f, indent=2)
                    f.close()
                    self.packet_sent_num += 1

                match self.msg['msgType']:
                    case 'resetEpisode':
                        should_keep_evaluating = True

                        if self.training_type == TrainingType.TRAINING:
                            self.episode_num += 1
                            self.debug_logs_dir = os.path.join(self.tensorboard_log, 'training', f'episode_{self.episode_num}')
                        else:
                            should_keep_evaluating = ((self.inference_test_num[1] + 1) < self.eval_inference_freq.frequency) if (self.eval_inference_freq.unit == TrainFrequencyUnit.EPISODE) else (
                                    self.eval_packet_sent < self.eval_inference_freq.frequency)

                            if should_keep_evaluating:
                                self.inference_test_num = (self.inference_test_num[0], self.inference_test_num[1] + 1)
                                self.debug_logs_dir = os.path.join(self.tensorboard_log, 'inference', f'run_{self.inference_test_num[0]}', f'episode_{self.inference_test_num[1]}')
                            else:
                                # This is to prevent writing any logs that are not ever going to be used, i.e. after an evaluation is complete, gymnasium enforces a reset that is never used
                                self.msg_discard = True

                        if (self.training_type == TrainingType.TRAINING) or should_keep_evaluating:
                            self.clean_and_create_debug_directories()
                            self.packet_received_num = 0
                            self.packet_sent_num = 0
                            self.image_num = 0
                    case 'inferenceEnd':
                        self.debug_logs_dir = os.path.join(self.tensorboard_log, 'training', f'episode_{self.episode_num}')
                        self.packet_received_num = len(os.listdir(os.path.join(self.debug_logs_dir, 'packets_received')))
                        self.packet_sent_num = len(os.listdir(os.path.join(self.debug_logs_dir, 'packets_sent')))
                        self.image_num = len(os.listdir(os.path.join(self.debug_logs_dir, 'images')))
                    case _:
                        pass

            if self.protocol == Protocol.UDP:
                self.sock.sendto(bytes(json_str, encoding='utf-8'), self.addr)
            else:
                self.conn.sendall(bytes(json_str, encoding='utf-8'))

            self.msg = None
            self.msg_event.clear()

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

    def send_world_state(self, world_state):
        self.set_msg({
            'msgType': 'setWorldState',
            'payload': {
                'objectOverrides': world_state
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

    def send_inference_start(self, eval_inference_freq):
        self.training_type = TrainingType.INFERENCE
        self.eval_inference_freq = eval_inference_freq
        self.set_msg({
            'msgType': 'inferenceStart',
            'payload': {
                'inferenceData': {
                    'evalInferenceFreq': self.eval_inference_freq.unit.value,
                    'nEvalCount': self.eval_inference_freq.frequency
                }
            }
        })

    def send_inference_end(self):
        self.training_type = TrainingType.TRAINING
        self.set_msg({
            'msgType': 'inferenceEnd',
            'payload': {}
        })

    def clean_and_create_debug_directories(self):
        clean_and_remake(self.debug_logs_dir)
        clean_and_remake(os.path.join(self.debug_logs_dir, 'packets_received'))
        clean_and_remake(os.path.join(self.debug_logs_dir, 'packets_sent'))
        clean_and_remake(os.path.join(self.debug_logs_dir, 'images'))
