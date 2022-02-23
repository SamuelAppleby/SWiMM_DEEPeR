# -*- coding:utf-8 -*-
"""This implements the skeleton for the server for the machine learning task"""

import socket
import os
import json
from datetime import datetime
import time

BUFF_SIZE = 4096  # 4 KiB

class PythonServer():
    """
    Handles messages from a single TCP client.
    """

    def __init__(self):

        self.saveToFile = False

    def write_image_to_file_incrementally(self, image):
        """
        Dumping the image to a continuously progressing file, just for debugging puroses
        """
        i = 0
        while os.path.exists("sample%s.jpeg" % i):
            i += 1
        with open("sample%s.jpeg" % i, "wb") as f:
            f.write(image)

    def relational_learning_model(self, image):
        return json.dumps({
            "forwardThrust": 1,
            "verticalThrust": 0,
            "yRotation": 0
        })

    def get_cam_config(self):
        return {
            "fov" : 100
        }

    def get_rover_config(self):
        return {
            "thrustPower" : 12
        }

    def get_server_config(self):
        jsonObj = {
            "serverConfig" : {
                "camConfig" : self.get_cam_config(),
                "roverConfig" : self.get_rover_config()
            }
        }
        return json.dumps(jsonObj)

    def server(self, host='127.0.0.1', port=60260):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            print("[+] Listening on {0}:{1}".format(host, port))
            sock.listen(5)

            while True:
                conn, addr = sock.accept()

                with conn:
                    currentTime = datetime.now().ctime()
                    total_data = []
                    print(
                        "[+] Connecting by {0}:{1} ({2})".format(addr[0], addr[1], currentTime))

                    data = b''       # binary data (not utf-8 nor ascii)

                    server_config = self.get_server_config()
                    conn.sendall(server_config.encode('utf-8'))
                    print("[+] Sent server config to {0}:{1}".format(addr[0], addr[1]))

                    while conn.fileno:
                        part = conn.recv(BUFF_SIZE)

                        if not part:
                            print("[-] Not Binary Image, Closing Socket")
                            conn.close()
                            break

                        data += part  # append the data

                        if not data:
                            print("[-] Not Received")
                            break

                        print("[+] Received", len(data))

                        if self.saveToFile:
                            self.write_image_to_file_incrementally(data)

                        actions = self.relational_learning_model(data)

                        conn.sendall(actions.encode('utf-8'))
                        #time.sleep(0.1)
                        print("[+] Sent action to {0}:{1}".format(addr[0], addr[1]))


#if __name__ == "__main__":
    #server()
