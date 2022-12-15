"""
This file contains all the run parameters and Gym environment parameters. This is different to the algorithm hyperparameters, 
which are stored in the yaml file associated with the chosen algorithm, in the 'hyperparams' folder.
"""

# The following are run parameters 

# Algorithm to import from Stable Baselines. CHOICES: 'sac', 'ddpg', 'ppo'
import os


ALGO = 'sac'

# If want to train on top of an existing model, provide filepath, else pass empty string
MODEL = ''

# Base filepath for saving outputs and logs
BASE_FILEPATH = 'gym_underwater' + os.sep + 'Logs'

# Whether or not to use Tensorboard NOTE: resulting event files are large
TB = True

# Whether to save the Monitor logs generate by the Stable Baselines Monitor wrapper NOTE: False still writes but to tmp
MONITOR = True

# Override log interval (default: -1, no change)
LOG_INTERVAL = -1

# Verbose mode (0: no output, 1: INFO)
VERBOSE = 1


# The following are environment parameters

# Observation type. CHOICES: 'image', 'vector'
OBS = 'image'

# If using image obs, declare desired output size of scaling function
IMG_SCALE = (84,84,3)

# Used to express the optimal tracking position. This value is subtracted from the z-coordinate of the target's position 
# i.e. how far set back from the target do you want the AUV to position itself
OPT_D = 10

# At what radial distance should the episode terminate (FROM THE OPTIMAL TRACKING POSITION, NOT THE TARGET POSITION)
MAX_D = 30
