# -*- coding:utf-8 -*-

import socket
import os
import json
from datetime import datetime
import time
from threading import Thread

BUFF_SIZE = 4096  # 4 KiB

class PythonServer():
    """
    Handles message passing with a single TCP client.
    Here, Python backend is the server and Unity frontend is the client.
    """

    def __init__(self, address, handler):

        # hold onto the handler
        self.handler = handler

        # some variable initialisations
        self.saveToFile = False
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
        while os.path.exists("sample%s.jpeg" % i):
            i += 1
        with open("sample%s.jpeg" % i, "wb") as f:
            f.write(image)
    
    def connect(self, host='127.0.0.1', port=60260):

        # create socket and associate the socket with local address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        print("[+] Listening on {0}:{1}".format(host, port))
        self.sock.listen(5)

        try:
            # wait for connection request
            conn, self.addr = self.sock.accept()
            currentTime = datetime.now().ctime()
            print("[+] Connecting by {0}:{1} ({2})".format(self.addr[0], self.addr[1], currentTime))
        except ConnectionRefusedError:
            raise (
                Exception(
                    "Could not connect to server."
                )
            )

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

    def proc_msg(self, conn):

        # binary data (not utf-8 nor ascii)
        data = b''       

        # conn.fileno being used to detect if socket has detached
        while self.do_process_msgs and conn.fileno:

            # receive 
            part = conn.recv(BUFF_SIZE)

            # check
            if not part:
                print("[-] Not Binary Image")
                self.stop()
                break
            
            # append
            data += part  

            if not data:
                print("[-] Not Received")
                break

            print("[+] Received", len(data))

            # for testing purposes, set to False by default
            if self.saveToFile:
                self.write_image_to_file_incrementally(data)

            # pass data to handler
            self.handler.on_recv_message(data)

            # wait for handler to point something to self.msg variable dedicated to outgoing messages
            while self.msg is None:
                time.sleep(1.0 / 120.0)
            # send 'reply' to client
            conn.sendall(self.msg.encode('utf-8'))
            print("[+] Sent action to {0}:{1}".format(self.addr[0], self.addr[1]))

                
            













#     def server(self, host='127.0.0.1', port=60260):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#             sock.bind((host, port))
#             print("[+] Listening on {0}:{1}".format(host, port))
#             sock.listen(5)

#             while True:
#                 conn, addr = sock.accept()

#                 with conn:
#                     currentTime = datetime.now().ctime()
#                     total_data = []
#                     print(
#                         "[+] Connecting by {0}:{1} ({2})".format(addr[0], addr[1], currentTime))

#                     data = b''       # binary data (not utf-8 nor ascii)

#                     server_config = self.get_server_config()
#                     conn.sendall(server_config.encode('utf-8'))
#                     print("[+] Sent server config to {0}:{1}".format(addr[0], addr[1]))

#                     while conn.fileno:
#                         part = conn.recv(BUFF_SIZE)

#                         if not part:
#                             print("[-] Not Binary Image, Closing Socket")
#                             conn.close()
#                             break

#                         data += part  # append the data

#                         if not data:
#                             print("[-] Not Received")
#                             break

#                         print("[+] Received", len(data))

#                         if self.saveToFile:
#                             self.write_image_to_file_incrementally(data)

#                         actions = self.relational_learning_model(data)

#                         conn.sendall(actions.encode('utf-8'))
#                         #time.sleep(0.1)
#                         print("[+] Sent action to {0}:{1}".format(addr[0], addr[1]))


# if __name__ == "__main__":
#     server()
