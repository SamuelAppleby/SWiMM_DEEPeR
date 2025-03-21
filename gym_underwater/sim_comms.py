import base64
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import time
import socket
import threading
from typing import Tuple

import numpy as np
import math

from gymnasium import Space
from jsonschema.validators import validate
from datetime import datetime

import cmvae_utils.dataset_utils
from gym_underwater.constants import ALPHA, SMOOTHNESS_THRESHOLD, SMOOTHNESS_PENALTY, MAX_REWARD
from gym_underwater.enums import EpisodeTerminationType, TrainingType, ObservationType, RenderType
from gym_underwater.mathematics import calc_metrics, normalized_exponential_impact, normalized_natural_log_impact, normalized_absolute_difference


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~#
def run_executable(path, args):
    subprocess.run([path] + args)


def launch_simulation(args, linux=False) -> threading.Thread:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'builds', 'linux', 'SWiMM_DEEPeR.x86_64') if linux else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'builds', 'windows', 'SWiMM_DEEPeR.exe')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Executable not found at {path}")

    thread = threading.Thread(target=run_executable, args=(path, args))
    thread.start()
    return thread


def process_and_validate_configs(dir_map):
    arr = []
    for conf_dir, schema_dir in dir_map.items():
        assert os.path.isfile(conf_dir) and os.path.isfile(schema_dir)

        with open(conf_dir) as file_conf:
            conf_json = json.load(file_conf)

            with open(schema_dir) as file_schema:
                schema_json = json.load(file_schema)
                validate(instance=conf_json, schema=schema_json)
                arr.append(conf_json)

    return arr


def clean_and_remake(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


PERIOD = 120


class ClientDisconnectedException(Exception):
    pass


class UnitySimHandler:
    def __init__(self,
                 img_res: Tuple[int, int, int],
                 tensorboard_log: str,
                 ip: str,
                 port: int,
                 training_type: TrainingType,
                 cmvae,
                 action_space: Space,
                 render: RenderType,
                 seed: int,
                 compute_stats: bool):
        self.compute_stats = compute_stats
        self.weights = np.array([0.03, 0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25])
        self.current_info = {
            'episode_num': -1,
            'a_error': -1.0,
            'd_error': -1.0,
            'out_of_view': 0,
            'maximum_distance': 0,
            'target_collision': 0,
            'a_smoothness_error': -1.0,
            'd_smoothness_error': -1.0
        }
        self.interval = 1 / PERIOD
        self.sim_ready = False
        self.last_obs = None
        self.rover_info = None
        self.target_info = None
        self.msg_queue = queue.Queue()
        self.img_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.img_res = img_res
        self.debug_logs = tensorboard_log is not None
        self.previous_actions = queue.Queue(maxsize=10)
        self.action_space = action_space

        self.fns = {
            'clientReady': self.on_client_ready,
            'onTelemetry': self.on_telemetry
        }

        self.episode_num = 0
        self.packet_received_num = 0
        self.packet_sent_num = 0
        self.image_num = 0
        self.cmvae = cmvae

        conf_arr = process_and_validate_configs({
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'server_config.json'): os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'schemas',
                                                                                                                               'server_config_schema.json')
        })

        self.server_config = conf_arr.pop()
        self.opt_d = self.server_config['payload']['serverConfig']['envConfig']['optD']
        self.max_d = self.server_config['payload']['serverConfig']['envConfig']['maxD']

        self.training_type = training_type

        exe_args = ['-ip', ip, '-port', str(port), '-modeServerControl', '-trainingType', str(training_type), '-seed', str(seed)]

        match render:
            case RenderType.HUMAN:
                pass
            case RenderType.NONE:
                exe_args.append('-batchmode')
            case _:
                pass

        self.sock = None
        self.addr = None
        self.conn = None
        self.threads = []

        self.address = (ip, port)

        self.thread_exe = launch_simulation(args=exe_args)

        self.read_write_thread = self.connect(*self.address)

        self.debug_logs_dir = None

        if self.debug_logs:
            parent_type = 'training' if self.training_type == TrainingType.TRAINING else 'inference'
            self.debug_logs_dir = os.path.join(tensorboard_log, 'network', parent_type, f'{self.address[0]}_{self.address[1]}', f'episode_{self.episode_num}')
            clean_and_remake(self.debug_logs_dir)
            self.clean_and_create_debug_directories()
            exe_args.append('-debugLogs')

        self.read_write_thread.start()

    def connect(self, host='127.0.0.1', port=60260) -> threading.Thread:
        """
        Open a tcp/udp socket
        """
        # create socket and associate the socket with local address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock.bind((host, port))
        print(f'Listening on {host}:{port}')

        while self.conn is None:
            try:
                self.sock.listen()
                self.conn, self.addr = self.sock.accept()
            except ConnectionRefusedError as refuse_error:
                print('Connection refused:', refuse_error)

        print(f'Connecting by {self.addr[0]}:{self.addr[1]} ({datetime.now().ctime()})')
        return threading.Thread(target=self.continue_read_write, daemon=True)

    def set_msg(self, msg):
        self.msg_queue.put(msg)

    def reset(self):
        self.previous_actions.queue.clear()
        self.last_obs = None
        self.rover_info = None
        self.target_info = None
        self.set_msg({
            'msgType': 'resetEpisode',
            'payload': {}
        })

    def observe(self,
                obs: ObservationType):
        while not self.img_queue.empty() and self.read_write_thread.is_alive():
            time.sleep(self.interval)

        if self.cancel_event.is_set():
            self.read_write_thread.join()
            sys.exit(0)

        img_array = self.img_queue.get()

        # Sanity check, we should NEVER have two identical images following each other, as that violates rendering and Newtonian physics
        # if self.last_obs is not None:
        #     assert not np.array_equal(img_array, self.last_obs)

        self.last_obs = img_array

        info = {
            'rover': self.rover_info,
            'target': self.target_info
        }

        # for vector obs training run, overwrite image observation with vector obs
        # observation and self.last_obs left in because orchestrates above while loop which is making Python server wait for next message from client
        match obs:
            case ObservationType.VECTOR:
                img_array = np.array([info['rover']['pos'][0], info['rover']['pos'][1], info['rover']['pos'][2], info['rover']['fwd'][0], info['rover']['fwd'][1], info['rover']['fwd'][2],
                                      info['target']['pos'][0], info['target']['pos'][1], info['target']['pos'][2], info['target']['fwd'][0], info['target']['fwd'][1], info['target']['fwd'][2]])
            # if vae has been passed, raw image observation encoded to latent vector
            case ObservationType.CMVAE:
                # add a dimension on the front so that has the shape (N, vae_res, vae_res, 3) that network expects
                img_array = np.expand_dims(img_array, axis=0)

                # set latent vector as observation
                img_array, _, _ = self.cmvae.encode(img_array)
                img_array = img_array.numpy()

        raw_d, a = calc_metrics(np.array(info['rover']['pos']), np.array(info['rover']['fwd']), np.array(info['target']['pos']))
        d_from_opt = math.fabs(raw_d - self.opt_d)

        # a_out_of_bounds only considers the center of the dolphin's mesh and should not be regarded as exact
        reward, d_out_of_bounds, a_out_of_bounds = self.calc_reward(d_from_opt, a)

        overs = []

        # During inference, we still allow TimeLimit truncation but want to simulate the real-world inference, so no early reset
        if self.training_type == TrainingType.TRAINING or self.compute_stats:
            overs = self.determine_episode_over(d_out_of_bounds)

            # If recording final model metrics, ignore the first observation's information as this has come from a reset, not a step
            if self.compute_stats and not self.previous_actions.empty():
                self.current_info['a_error'] = a
                self.current_info['d_error'] = d_from_opt

                for over in overs:
                    match over:
                        case EpisodeTerminationType.TARGET_OUT_OF_VIEW:
                            self.current_info['out_of_view'] = 1 if (self.current_info['out_of_view'] == 0) else 1
                        case EpisodeTerminationType.MAXIMUM_DISTANCE:
                            self.current_info['maximum_distance'] = 1 if (self.current_info['maximum_distance'] == 0) else 1
                        case EpisodeTerminationType.TARGET_COLLISION:
                            self.current_info['target_collision'] = 1 if (self.current_info['target_collision'] == 0) else 1
                        case _:
                            pass

            if self.training_type == TrainingType.INFERENCE:
                overs = []

        info.update({
            'dist': raw_d,
            'ang_error': a,
            'episode_termination_type': overs
        })

        # During training, guarantee that reward is between -1 and 1 if not over
        if (self.training_type == TrainingType.TRAINING) and (len(overs) == 0):
            assert (-MAX_REWARD <= reward <= MAX_REWARD), f'We cannot allow a reward outside of the range [-{MAX_REWARD}, {MAX_REWARD}]'

        img_array = self.get_augmented_state(img_array)
        return img_array, reward, len(overs) > 0, False, info

    def calc_reward(self, d_from_opt, a):
        # scaling function producing value in the range [-1, 1] - distance and angle equal contribution
        distance_penalty, d_out_of_bounds = normalized_exponential_impact(diff=d_from_opt, max_diff=self.max_d, k=1)
        angle_penalty, a_out_of_bounds = normalized_natural_log_impact(diff=a, max_diff=ALPHA, k=1)

        # smoothness_penalty = 0
        # if self.previous_actions.qsize() > 1:
        #     action_list = list(self.previous_actions.queue)
        #     action_diff = normalized_absolute_difference(action_list[-1], action_list[-2], self.action_space)
        #
        #     smoothness_penalty_array = np.zeros_like(action_diff)
        #     smoothness_penalty_array[action_diff > SMOOTHNESS_THRESHOLD] = SMOOTHNESS_PENALTY
        #     smoothness_penalty = np.sum(smoothness_penalty_array)
        #
        # # Reward can be between [-1.5, 1.5]
        # return (MAX_REWARD - (distance_penalty + angle_penalty + smoothness_penalty)), d_out_of_bounds, a_out_of_bounds

        return (MAX_REWARD - (distance_penalty + angle_penalty)), d_out_of_bounds, a_out_of_bounds

    def determine_episode_over(self, d_out_of_bounds):
        overs = []
        if self.target_info['colliding']:
            print('[EPISODE TERMINATED] Due to collision with target')
            overs.append(EpisodeTerminationType.TARGET_COLLISION)
        if self.target_info['outOfView']:
            print('[EPISODE TERMINATED] Due to target out of view')
            overs.append(EpisodeTerminationType.TARGET_OUT_OF_VIEW)
        if d_out_of_bounds:
            print(f'[EPISODE TERMINATED] Too far from the optimum distance')
            overs.append(EpisodeTerminationType.MAXIMUM_DISTANCE)
        return overs

    def continue_read_write(self):
        self.threads = [
            threading.Thread(target=self.send_msg, daemon=True),
            threading.Thread(target=self.recv_msg, daemon=True)
        ]

        for t in self.threads:
            t.start()

        for t in self.threads:
            t.join()
            print('THREAD JOINED')

        self.disconnect()
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
                # Read the length prefix (4 bytes)
                length_prefix = self.conn.recv(4)
                if not length_prefix:
                    raise ClientDisconnectedException('Client closed the connection')

                # Unpack the length prefix as an unsigned integer (little-endian)
                msg_length = struct.unpack('<I', length_prefix)[0]

                data = self.conn.recv(msg_length)

                if not data or len(data) != msg_length:
                    raise ValueError(f'Incomplete message received. Expected {msg_length} bytes, received {len(data)} bytes.')

                json_str = data.decode('UTF-8')
                json_dict = json.loads(json_str)

                if self.debug_logs_dir is not None:
                    with open(os.path.join(self.debug_logs_dir, 'packets_received', f'step_{self.packet_received_num}.json'), 'w', encoding='utf-8') as f:
                        json.dump(json_dict, f, indent=2)
                        f.close()
                        self.packet_received_num += 1

                msg_type = json_dict['msgType']
                assert msg_type is not None and msg_type in self.fns, f'Unknown message type {msg_type}'
                self.fns[msg_type](json_dict['payload'])

            except Exception as e:
                print('Stop receive:', e)
                self.cancel_event.set()

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

        self.img_queue.put(image)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Outgoing Communications ~~~~~~~~~~~~~~~~~~~~~~~~~#
    def send_msg(self):
        """
        Continue to process messages from each connected client
        """
        while not self.cancel_event.is_set():
            try:
                while self.msg_queue.empty() and not self.cancel_event.is_set():
                    time.sleep(self.interval)

                if self.cancel_event.is_set():
                    continue

                msg = self.msg_queue.get()
                assert msg is not None

                json_str = json.dumps(msg)
                json_bytes = json_str.encode('utf-8')
                json_length = struct.pack('<I', len(json_bytes))  # Little endian

                assert bytes(json_str, encoding='utf-8') == json_bytes

                if self.debug_logs_dir is not None:
                    with open(os.path.join(self.debug_logs_dir, 'packets_sent', f'step_{self.packet_sent_num}.json'), 'w', encoding='utf-8') as f:
                        json.dump(msg, f, indent=2)
                        f.close()
                        self.packet_sent_num += 1

                    match msg['msgType']:
                        case 'resetEpisode':
                            self.episode_num += 1
                            self.debug_logs_dir = self.debug_logs_dir[:self.debug_logs_dir.rfind('_') + 1] + str(self.episode_num)
                            self.clean_and_create_debug_directories()
                            self.packet_received_num = 0
                            self.packet_sent_num = 0
                            self.image_num = 0
                        case _:
                            pass

                self.conn.sendall(json_length + json_bytes)

            except Exception as e:
                print('Stop send:', e)
                self.cancel_event.set()

    def send_action(self, action):
        if not self.previous_actions.empty():
            absol = np.abs(self.previous_actions.queue - action)
            weighted_abs = np.multiply(absol, self.weights[self.weights.size - self.previous_actions.qsize():, np.newaxis])
            weight_sum = np.sum(weighted_abs, axis=0)
            norm_sum = weight_sum / (self.action_space.high - self.action_space.low)
            self.current_info['a_smoothness_error'] = norm_sum[1]
            self.current_info['d_smoothness_error'] = norm_sum[0]
        else:
            self.current_info['episode_num'] += 1

            self.current_info = {
                'episode_num': self.current_info['episode_num'],
                'a_error': -1.0,
                'd_error': -1.0,
                'out_of_view': 0,
                'maximum_distance': 0,
                'target_collision': 0,
                'a_smoothness_error': 0.0,
                'd_smoothness_error': 0.0
            }

        if self.previous_actions.full():
            self.previous_actions.get()  # Remove the oldest action
        self.previous_actions.put(action)  # Add the new action

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
        self.thread_exe.join()
        self.read_write_thread.join()

    def get_augmented_state(self, state):
        # Flatten the list of previous actions
        previous_actions_np = np.array(list(self.previous_actions.queue)).flatten()

        if len(previous_actions_np) < 20:
            padding = np.zeros(20 - len(previous_actions_np))
            previous_actions_np = np.concatenate([previous_actions_np, padding])

        return np.concatenate([state.flatten(), previous_actions_np]).reshape(1, -1)
