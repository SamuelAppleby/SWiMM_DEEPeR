"""
This file contains all hardcoded variables, including anything that is static 
and anything that is a decision for the user e.g. parameters of the sim 
that are changeable from the Python end, or parameters of the Gym environment.
This discludes RL algorithm hyperparameters, which are stored in a .yml file 
within the repo's 'hyperparams' folder. 
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~ Networking ~~~~~~~~~~~~~~~~~~~~~~~~~#

HOST = '127.0.0.1'
PORT = 60260

#~~~~~~~~~~~~~~~~~~~~~~~~~ Unity Sim ~~~~~~~~~~~~~~~~~~~~~~~~~#

SERVER_CONF = {
    "msgType": "process_server_config",
    "payload": {
        "camConfig": {
            "fov": 50
        },
        "envConfig": {
            "fogStart": 5
        }
    }
}

GLOBAL_MSG_TEMPLATE = {
    "msgType": "global_message",
    "payload": {

    }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~ Gym Environment ~~~~~~~~~~~~~~~~~~~~~~~~~#

OPT_D = 10
MAX_D = 30

#~~~~~~~~~~~~~~~~~~~~~~~~~ Debugging ~~~~~~~~~~~~~~~~~~~~~~~~~#

SAVE_IMAGES = False
SAVE_IMAGES_TO = "../../debug_images/"
