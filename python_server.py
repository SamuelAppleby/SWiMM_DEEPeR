# -*- coding:utf-8 -*-
"""This implements the skeleton for the server for the machine learning task"""

import socket
import os
from datetime import datetime



def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''       # binary data (not utf-8 nor ascii)
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part # append the data
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data

#<<<<<<< HEAD
def write_image_to_file_incrementally(image):
    """
    Dumping the image to a continuously progressing file, just for debugging puroses
    """
    i = 0
    while os.path.exists("sample%s.jpeg" % i):
        i += 1
    with open("sample%s.jpeg" % i, "wb") as f:
        f.write(image)

def relational_learning_model(image):
    # If you want, you can also dump the image into a file, just for testing that it works, forsooth!
    write_image_to_file_incrementally(image)
    #=======
    #def relational_learning_model(image)
    #    """
    #    This metod takes as an input the binary representation of a jpg (PNG is too slow to dump) and returns the set of actions 
    #    encoded as a single string that need to be decoded by C#
    #    """
    #>>>>>>> de3b4f34202d4cb271cae931834fa69dc450c759
    # returns the actions to be performed by the rover
    # See the unity source code for some specs
    return "F"

# address and port is arbitrary
def server(host='127.0.0.1', port=60260):
  # create socket
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind((host, port))
    print("[+] Listening on {0}:{1}".format(host, port))
    sock.listen(5)
    # permit to access
    conn, addr = sock.accept()

    with conn as c:
      # display the current time
      time = datetime.now().ctime()
      total_data = []
      print("[+] Connecting by {0}:{1} ({2})".format(addr[0], addr[1], time))

      while True:
        binary_image = recvall(conn)

        if not binary_image:
          print("[-] Not Received")
          break

        # the image is completely received
        print("[+] Received", len(binary_image))
        # TODO: do something with the image. E.g., send it to the
        # relational learning model and return the set of actions
        # that the simulated rover has to perform
        actions = relational_learning_model(binary_image)
        # Encode the string via utf-8, and send the result as a byte array
        c.sendall(actions.encode('utf-8'))
        print("[+] Sending to {0}:{1}".format(addr[0], addr[1]))

if __name__ == "__main__":
  server()
