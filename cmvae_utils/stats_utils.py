'''
file: stats_utils.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def calculate_state_stats(predictions, poses, output_dir):

    f_path = os.path.join(output_dir, 'prediction_stats.txt')
    f = open(f_path, 'w')

    # display averages
    mean_pred = np.mean(predictions, axis=0)
    mean_pose = np.mean(poses, axis=0)
    print('Means (prediction, GT) : R({} , {}) Theta({} , {}) Psi({} , {})'.format( 
        mean_pred[0], mean_pose[0], mean_pred[1], mean_pose[1], mean_pred[2], mean_pose[2]), file=f) 
    # display mean absolute error
    abs_diff = np.abs(predictions-poses)
    mae = np.mean(abs_diff, axis=0)
    #mae[1:] = mae[1:] * 180/np.pi
    print('MAE : R({}) Theta({}) Psi({})'.format(mae[0], mae[1], mae[2]), file=f) 
    # display standard deviation of error
    std = np.std(abs_diff, axis=0) / np.sqrt(abs_diff.shape[0])
    #std[1:] = std[1:] * 180 / np.pi
    print('Standard error: R({}) Theta({}) Psi({})'.format(std[0], std[1], std[2]), file=f) 
    # display max errors
    max_diff = np.max(abs_diff, axis=0)
    print('Max error : R({}) Theta({}) Psi({})'.format(max_diff[0], max_diff[1], max_diff[2]), file=f) 

    f.close() 

    fig, axs = plt.subplots(1, 3, tight_layout=True)
    weights = np.ones(len(abs_diff[:, 0]))/len(abs_diff[:, 0])

    theta_max = (90.0*0.7)/2.0

    axs[0].hist(abs_diff[:, 0], bins=30, range=(3.0,30.0), weights=weights, density=False) #2.0
    axs[1].hist(abs_diff[:, 1], bins=30, range=(-theta_max, theta_max), weights=weights, density=False) 
    axs[2].hist(abs_diff[:, 2], bins=50, range=(-90.0, 90.0), weights=weights, density=False) 

    for idx in range(3):
        axs[idx].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    axs[0].set_title(r'$r$')
    axs[1].set_title(r'$\phi$')
    axs[2].set_title(r'$\psi$')

    axs[0].set_xlabel('[m]')
    axs[1].set_xlabel(r'[deg]')
    axs[2].set_xlabel(r'[deg]')

    axs[0].set_ylabel('Error Density')

    fig.savefig(os.path.join(output_dir, 'state_stats_error_histograms.png'))

    plt.show()

def calc_abs_yaw_stats(predictions, poses, output_dir):

    f_path = os.path.join(output_dir, 'abs_stats.txt')
    f = open(f_path, 'w')

    # see how model does at predicting yaw regardless of sign
    abs_psi_predictions = []
    for psi in predictions[2]:
        abs_psi_predictions.append(np.abs(psi))
    abs_psi_gt = []
    for psi in poses[2]:
        abs_psi_gt.append(np.abs(psi))
    abs_psi_predictions = np.array(abs_psi_predictions)
    abs_psi_gt = np.array(abs_psi_gt)
    new_abs_diff = np.abs(abs_psi_predictions-abs_psi_gt)
    new_mae = np.mean(new_abs_diff, axis=0)
    new_std = np.std(new_abs_diff, axis=0) / np.sqrt(new_abs_diff.shape[0])
    new_max_diff = np.max(new_abs_diff, axis=0)
    print('If take absolute yaw: mae = {}, std = {}, max = {}'.format(new_mae, new_std, new_max_diff), file=f)

    f.close()

