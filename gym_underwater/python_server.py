# -*- coding:utf-8 -*-

import socket
import os
import json
from datetime import datetime
import time
from threading import Thread

BUFF_SIZE = 32768  # 32 KiB

class PythonServer():
    """
    Handles message passing with a single TCP client.
    Here, Python backend is the server and Unity frontend is the client.
    """

    def __init__(self, address, handler):

        # hold onto the handler
        self.handler = handler

        # some variable initialisations
        self.do_process_msgs = False
        self.save_to_file = True
        self.msg = None
        self.th = None

        # establishing connection
        self.connect(*address)
        # let handler know when connection has been made
        self.handler.on_connect(self)

    def write_image_to_file_incrementally(self, image):
        """
        Dumping the image to a continuously progressing file, just for debugging purposes
        """
        i = 0
        while os.path.exists(f"sample{i}.jpeg"):
            i += 1
        with open(f"sample{i}.jpeg", "wb") as f:
            f.write(image)
    
    def send_server_config(self, conn):
        """
        Send server config to client
        """

        server_conf = {
            "msg_type": "server_config",
            "payload": {
                "cam_config": {
                    "fov": 50
                },
                "env_config" : {
                    "fogStart" : 20
                }
            }
        }

        server_str = json.dumps(server_conf)
        conn.sendall(bytes(server_str,encoding="utf-8"))

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
            conn, self.addr = self.sock.accept()
            current_time = datetime.now().ctime()
            print(f"[+] Connecting by {self.addr[0]}:{self.addr[1]} ({current_time})")
        except ConnectionRefusedError as refuse_error:
            raise (Exception("Could not connect to server.")) from refuse_error
        
        self.send_server_config(conn)

        # the remaining network related code, receiving data and sending data, is ran in a thread
        self.do_process_msgs = True
        self.th = Thread(target=self.proc_msg, args=(conn,), daemon=True)
        self.th.start()

    def stop(self):
        """
        Signal proc_msg loop to stop, wait for thread to finish, close socket, and tell handler
        """
        self.do_process_msgs = False
        if self.th is not None:
            self.th.join()
        if self.sock is not None:
            print("[-] Closing socket")
            self.sock.close()
            self.handler.on_disconnect()

    def process_camera_image(self, payload):
        """
        Encode image, send for learning
        """
        image = payload["jpgImage"]
        # b = bytearray(image)

        #  # for testing purposes, set to False by default
        # if self.save_to_file:
        #     self.write_image_to_file_incrementally(b)

        # pass data to handler
        self.handler.on_recv_message(image)

    def proc_msg(self, conn):
        """
        Continue to process messages from each connected client
        """
        # conn.fileno being used to detect if socket has detached
        while self.do_process_msgs and conn.fileno:
            # receive
            part = conn.recv(1024 * 32)
            if not part:
                print("[-] Not Binary Image")
                self.stop()
                break
            
            print("[+] Received", len(part))
            my_json = part.decode('UTF-8')
            json_dict = json.loads(my_json)
            getattr(self, json_dict["msg_type"])(json_dict["payload"])

            # wait for handler to point something to self.msg variable dedicated to outgoing messages
            while self.msg is None:
                print("i am waiting")
                time.sleep(1.0 / 120.0)
            # send 'reply' to client
            conn.sendall(self.msg.encode('utf-8'))
            print(f"[+] Sent action to {self.addr[0]}:{self.addr[1]}")
