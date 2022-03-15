"""
This file will contain anything that is a design choice, 
including parameters of the sim that are changeable from the Python end,
network parameters, and parameters of the gym environment,
so that user only needs to make edits in one place.
RL algorithm hyperparameters will be in a .yml file within the hyperparams folder 
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~ Networking ~~~~~~~~~~~~~~~~~~~~~~~~~#

HOST = '127.0.0.1'
PORT = 60260

#~~~~~~~~~~~~~~~~~~~~~~~~~ Unity Sim ~~~~~~~~~~~~~~~~~~~~~~~~~#

FOV = 50
FOGSTART = 20

#~~~~~~~~~~~~~~~~~~~~~~~~~ Gym Environment ~~~~~~~~~~~~~~~~~~~~~~~~~#



#~~~~~~~~~~~~~~~~~~~~~~~~~ Debugging ~~~~~~~~~~~~~~~~~~~~~~~~~#

SAVE_IMAGES = False
SAVE_IMAGES_TO = "../../debug_images/"
