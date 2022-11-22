# -*- coding:utf-8 -*-

import socket
import json
from jsonschema import validate
from datetime import datetime
import time
from threading import Thread
import os
import shutil
from config import *


class PythonServer():
    """
    Handles message passing with a single TCP client.
    Python backend is the server and Unity frontend is the client.
    """

    def __init__(self, handler):

        # hold onto the handler
        self.handler = handler

        # some variable initialisations
        self.do_process_msgs = False
        self.msg = None
        self.th = None
        self.conn = None
        self.debug_config = None
        self.network_config = None
        self.server_config = None
        self.receive_buffer_size = None

        # process the debug config
        self.debug_config = self.process_and_validate_config(
            'Configs/data/server_debug_config.json', 'Configs/schemas/server_debug_config_schema.json')

        # process the network config
        self.network_config = self.process_and_validate_config(
            'Configs/data/network_config.json', 'Configs/schemas/network_config_schema.json')
        self.address = (
            self.network_config["host"], self.network_config["port"])
        self.receive_buffer_size = self.network_config["buffers"]["server_receive_buffer_size_kb"]

        # process the server config
        self.server_config = self.process_and_validate_config(
            'Configs/data/server_config.json', 'Configs/schemas/server_config_schema.json')

        # clean cache (old images, logs etc)
        self.clean_cache()

        # establishing connection
        self.connect(*self.address)

    def clean_cache(self):
        dirpath = self.debug_config["image_dir"]
        if os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

    def process_and_validate_config(self, conf_dir, schema_dir):
        f = open(conf_dir, "r")
        conf_json = json.load(f)

        f = open(schema_dir, "r")
        schema_json = json.load(f)

        validate(instance=conf_json, schema=schema_json)

        return conf_json

    def connect(self, host='127.0.0.1', port=60260):
        """
        Open a socket and listen to maximium of 5 connections
        """

        # create socket and associate the socket with local address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        print(f"[+] Listening on {host}:{port}")
        self.sock.listen(5)

        try:
            # wait for connection request
            self.conn, self.addr = self.sock.accept()
            current_time = datetime.now().ctime()
            print(
                f"[+] Connecting by {self.addr[0]}:{self.addr[1]} ({current_time})")
        except ConnectionRefusedError as refuse_error:
            raise (Exception("Could not connect to server.")) from refuse_error

        # let handler know when connection has been made
        self.handler.on_connect(self)
        self.handler.generate_server_config()
        self.send_msg()

        # the remaining network related code, receiving data and sending data, is ran in a thread
        self.do_process_msgs = True
        self.th = Thread(target=self.proc_msg, args=(self.conn,), daemon=True)
        self.th.start()

    def stop(self):
        """
        Signal proc_msg loop to stop, wait for thread to finish, close socket, and tell handler
        """
        self.do_process_msgs = False

        new_msg = {
            "msgType": "end_simulation",
            "payload": {
            }
        }

        self.msg = json.dumps(new_msg)

        if self.th is not None:
            self.th.join()
        if self.sock is not None:
            print("[-] Closing socket")
            self.sock.close()
            self.handler.on_disconnect()

    def proc_msg(self, conn):
        """
        Continue to process messages from each connected client
        """
        # conn.fileno being used to detect if socket has detached
        while self.do_process_msgs and conn.fileno:

            # receive packets
            part = conn.recv(1024 * self.receive_buffer_size)

            if not part:
                print("[-] Not Binary Image")
                self.stop()
                break

            # unpack and send json message onto handler
            my_json = part.decode('UTF-8')
            print("I HAVE RECEIVED: " + my_json)
            json_dict = json.loads(my_json)
            self.handler.on_recv_message(json_dict)

            # wait for handler to point something to self.msg variable dedicated to outgoing messages
            # Sam A. Talk to Kirsten about this, potentially dangerous
            while self.msg is None and self.do_process_msgs:
                time.sleep(1.0 / 120.0)

            # send 'reply' to client
            self.send_msg()
            #print(f"[+] Sent action to {self.addr[0]}:{self.addr[1]}")

    def send_msg(self):
        print(self.msg.encode('utf-8'))
        self.conn.sendall(self.msg.encode('utf-8'))
        self.msg = None
