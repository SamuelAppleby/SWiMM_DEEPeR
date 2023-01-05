import time
import datetime
import os
import cv2
import csv
import numpy as np
from gym_underwater.gym_env import UnderwaterEnv
from cmvae_utils import geom_utils 

##################### USER SETTINGS #######################################################

# set the number of images wish to generate
num_samples = 104

# set the filepath where wish to save images to 
dataset_path = "Logs/vae_data"

############################################################################################ 

begin_time = datetime.datetime.now()

# Make image directory if doesn't already exist
os.makedirs(dataset_path, exist_ok=True)

# append a filename to the end of dataset path so have full path when it comes to writing state data to csv
csv_path = os.path.join(dataset_path, 'state_data.csv')

# Determine where to start image counter from if already images in folder
images_idx = [int(im.split('.png')[0]) for im in os.listdir(dataset_path) if im.endswith('.png')]
if len(images_idx) > 0:
    current_idx = max(images_idx) + 1 # the + 1 is so don't overwrite last existing image in the folder
else:
    current_idx = 1 # starting from 1 not 0 so matches csv row numbers

######################################################################################################################################
# FOR ALL REFERENCES TO X, Y AND Z IN THIS SCRIPT, X IS HORIZONTAL, Y IS FORWARD/BACK, AND Z IS VERTICAL, TO BE CONSISTENT WITH SCIPY
# ANYTHING COMING FROM OR GOING TO SIM WILL REQUIRE SWITCHING THE Y AND Z VALUE, AS UNITY DEFINES THE VERTICAL AXIS AS Y
######################################################################################################################################

# ranges for first moving rover around world 
ROVER_X_RANGE = [-30, 30] # world x coord
ROVER_Y_RANGE = [-30, 30] # world y coord
ROVER_YAW_RANGE = [-90, 90] # yaw value of Eular in degrees

# ranges for our 'states' that the CMVAE will learn to predict - radial distance (r), azimuth (theta), and yaw (psi) (all relative to rover, not absolute)
R_RANGE = [5, 30]  # in meters - any less than 3 causes objects to collide and spin up in air - at 30 target is still visible as small dot
CAM_FOV = 90.0*0.7  # in degrees -- multiplier is because of cone vs. square
alpha = (CAM_FOV/180.0*np.pi/2.0) # alpha is half of fov angle, in radians
THETA_RANGE = [-alpha, alpha] # radians required for polar translation, will convert to degrees before logging
PSI_RANGE = [-90, 90] # in degrees

env = UnderwaterEnv()

for _ in range(num_samples):

    # sample position and rotation for rover (world coords i.e. relative to origin)
    rover_pos_x = geom_utils.randomSample(ROVER_X_RANGE)
    rover_pos_y = geom_utils.randomSample(ROVER_Y_RANGE)
    rover_pos_z = -4 # hard coding the vertical position whilst restricting to 2D
    rover_yaw = geom_utils.randomSample(ROVER_YAW_RANGE)

    # express sampled yaw as a quarternion
    rover_Q = geom_utils.get_yaw_as_Q(rover_yaw)
    
    # send instruction off to sim to move tracking car 
    env.handler.send_object_pos('rover', rover_pos_x, rover_pos_y, rover_pos_z, rover_Q)

    # sample a value for radial distance (r) and azimuth angle (theta) NB: no phi (elevation) required when restricting to 2D
    r = geom_utils.randomSample(R_RANGE)
    theta_rel = geom_utils.randomSample(THETA_RANGE)

    # with the rover as the origin, generate (relative) position for dolphin using r and theta
    dolphin_rel_pos_x, dolphin_rel_pos_y = geom_utils.polarTranslation(r, theta_rel)
    dolphin_rel_pos = [dolphin_rel_pos_x, dolphin_rel_pos_y, 0] # both objects at same depth so at same elevation i.e. z_relative is zero

    # generate the dolphin's position in world coordinates
    dolphin_pos_world = geom_utils.convert_t_body_2_world(dolphin_rel_pos, rover_pos_x, rover_pos_y, rover_pos_z, rover_Q)

    # rotate the dolphin an amount relative to rover's rotation
    psi_rel = geom_utils.randomSample(PSI_RANGE)
    dolphin_yaw = rover_yaw + psi_rel

    # express yaw as a quarternion 
    dolphin_Q = geom_utils.get_yaw_as_Q(dolphin_yaw)

    # send instruction off to sim to move dolphin to this position and orientation
    env.handler.send_object_pos('dolphin', dolphin_pos_world[0], dolphin_pos_world[1], dolphin_pos_world[2], dolphin_Q) 

    time.sleep(0.3) # needed or else camera image is rendered before dolphin has moved

    # get image from rover's onboard camera
    image = env.handler.image_array

    # save image to file
    cv2.imwrite("{}/{}.png".format(dataset_path, current_idx), image) 

    # save state information to csv - should have same row number as the filename of the image
    row = [r, theta_rel/np.pi*180.0, psi_rel]
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)
        csv_file.close()

    if current_idx % 100 == 0:
        print('Num samples: {}'.format(current_idx))

    current_idx += 1
    
    #time.sleep(3) # sleep here is just so that sim runs slow enough to observe when testing

print("TIME TAKEN: ", datetime.datetime.now() - begin_time)

# exit scene? need to implement send_exit_scene message in sim_comms.py and then use here and at the end of train.py

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()