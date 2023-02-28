# -*- coding:utf-8 -*-

import socket
import json
from enum import Enum

from jsonschema import validate
from datetime import datetime
import time
from threading import Thread
import os
import shutil


class Protocol(Enum):
    UDP = 0
    TCP = 1


protocol_mapping = {
    "udp": Protocol.UDP,
    "tcp": Protocol.TCP
}


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


class PythonServer:
    """
    Handles message passing with a single TCP client.
    Python backend is the server and Unity frontend is the client.
    """

    def __init__(self, handler):

        # hold onto the handler
        self.sock = None
        self.addr = None
        self.handler = handler

        # some variable initialisations
        self.do_process_msgs = False
        self.msg = None
        self.th = None
        self.conn = None
        self.network_config = None
        self.protocol = None
        self.server_config = None
        self.action_buffer_size = None
        self.observation_buffer_size = None
        self.episode_num = -1
        self.action_num = 0

        conf_arr = process_and_validate_configs({
            os.path.abspath(os.path.join(os.pardir, 'Configs', 'json', 'network_config.json')): os.path.abspath(os.path.join(os.pardir, 'Configs', 'schemas', 'network_config_schema.json')),
            os.path.abspath(os.path.join(os.pardir, 'Configs', 'json', 'server_config.json')):  os.path.abspath(os.path.join(os.pardir, 'Configs', 'schemas', 'server_config_schema.json'))
        })

        self.server_config = conf_arr.pop()
        self.network_config = conf_arr.pop()
        self.protocol = protocol_mapping[self.network_config["protocol"]]
        self.address = (self.network_config["host"], self.network_config["port"])
        self.observation_buffer_size = self.network_config["buffers"]["channel_observation"]

        if self.handler.debug_logs:
            self.handler.clean_and_create_debug_directories()

        self.connect(*self.address)

    def connect(self, host='127.0.0.1', port=60260):
        """
        Open a udp socket
        """

        # create socket and associate the socket with local address
        print('Protocol' + str(self.protocol))

        if self.protocol == Protocol.UDP:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.sock.bind((host, port))
        print(f"[+] Listening on {host}:{port}")

        # the remaining network related code, receiving json and sending json, is ran in a thread
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
                break

            # unpack and send json message onto handler
            json_str = data.decode('UTF-8')
            # print('Received: {}'.format(json_str))
            json_dict = json.loads(json_str)

            if self.handler.debug_logs:
                with open(os.path.abspath(os.path.join(self.handler.packets_received_dir, 'episode_' + str(self.episode_num) + '_observation_' + str(json_dict['payload']['obsv_num']) + '.json')), 'w',
                          encoding='utf-8') as f:
                    json.dump(json_dict, f, ensure_ascii=False, indent=4)

            self.handler.on_recv_message(json_dict)

            # wait for handler to point something to self.msg variable dedicated to outgoing messages
            # print('Waiting to send message')
            while self.msg is None:
                time.sleep(1.0 / 120.0)

            # print('Sending: {}'.format(str(self.msg.encode('utf-8'))))
            self.msg['payload']['episode_num'] = self.episode_num
            self.msg['payload']['action_num'] = self.action_num
            json_str = json.dumps(self.msg)
            # print('Sending: {}'.format(json_str))

            if self.handler.debug_logs:
                if self.msg['msgType'] == "reset_episode":
                    self.handler.clean_and_create_debug_directories()
                else:
                    with open(os.path.abspath(os.path.join(self.handler.packets_sent_dir, 'episode_' + str(self.episode_num) + '_action_' + str(self.action_num) + '.json')), 'w', encoding='utf-8') as f:
                        json.dump(self.msg, f, ensure_ascii=False, indent=4)

            if self.protocol == Protocol.UDP:
                self.sock.sendto(bytes(json_str, encoding="utf-8"), self.addr)
            else:
                self.conn.sendall(bytes(json_str, encoding="utf-8"))

            self.action_num += 1
            self.msg = None

    def stop(self):
        """
        Signal proc_msg loop to stop, wait for thread to finish, close socket, and tell handler
        """
        self.do_process_msgs = False

        self.msg = {
            'msgType': 'end_simulation',
            'payload': {
            }
        }

        if self.th is not None:
            self.th.join()
        if self.sock is not None:
            print("[-] Closing socket")
            self.sock.close()
            self.handler.on_disconnect()
