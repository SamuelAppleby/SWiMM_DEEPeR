"""
This file contains all hardcoded variables, including anything that is static 
and anything that is a decision for the user e.g. parameters of the sim 
that are changeable from the Python end, or parameters of the Gym environment.
This discludes RL algorithm hyperparameters, which are stored in a .yml file 
within the repo's 'hyperparams' folder. 
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~ Unity Sim ~~~~~~~~~~~~~~~~~~~~~~~~~#

GLOBAL_MSG_TEMPLATE = {
    "msgType": "global_message",
    "payload": {

    }
}

#~~~~~~~~~~~~~~~~~~~~~~~~~ Gym Environment ~~~~~~~~~~~~~~~~~~~~~~~~~#

OPT_D = 10
MAX_D = 30
