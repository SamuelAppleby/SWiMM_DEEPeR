'''
file: dataset_utils.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import pandas as pd
import glob
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from natsort import natsorted


def convert_bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def convert_rgb2bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def normalize_state(pose):
    # normalization of ranges as used in image_gen.py to [-1, 1] range
    r_range = [3, 30]  # [0.1, 30]
    CAM_FOV = 90.0 * 0.7
    alpha = CAM_FOV / 2.0  # (CAM_FOV/180.0*np.pi/2.0)
    theta_range = [-alpha, alpha]  # [-90, 90]
    psi_range = [-90, 90]
    if len(pose.shape) == 1:
        # means that it's a 1D vector of velocities
        pose[0] = 2.0 * (pose[0] - r_range[0]) / (r_range[1] - r_range[0]) - 1.0
        pose[1] = 2.0 * (pose[1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
        pose[2] = 2.0 * (pose[2] - psi_range[0]) / (psi_range[1] - psi_range[0]) - 1.0
    elif len(pose.shape) == 2:
        # means that it's a 2D vector of velocities
        pose[:, 0] = 2.0 * (pose[:, 0] - r_range[0]) / (r_range[1] - r_range[0]) - 1.0
        pose[:, 1] = 2.0 * (pose[:, 1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
        pose[:, 2] = 2.0 * (pose[:, 2] - psi_range[0]) / (psi_range[1] - psi_range[0]) - 1.0
    else:
        raise Exception('Error in data format of V shape: {}'.format(pose.shape))
    return pose


def de_normalize_state(pose):
    # normalization of ranges as used in image_gen.py to [-1, 1] range
    r_range = [3, 30]  # [0.1, 30]
    CAM_FOV = 90.0 * 0.7
    alpha = CAM_FOV / 2.0  # (CAM_FOV/180.0*np.pi/2.0)
    theta_range = [-alpha, alpha]  # [-90, 90]
    psi_range = [-90, 90]
    if len(pose.shape) == 1:
        # means that it's a 1D vector of velocities
        pose[0] = (pose[0] + 1.0) / 2.0 * (r_range[1] - r_range[0]) + r_range[0]
        pose[1] = (pose[1] + 1.0) / 2.0 * (theta_range[1] - theta_range[0]) + theta_range[0]
        pose[2] = (pose[2] + 1.0) / 2.0 * (psi_range[1] - psi_range[0]) + psi_range[0]
    elif len(pose.shape) == 2:
        # means that it's a 2D vector of velocities
        pose[:, 0] = (pose[:, 0] + 1.0) / 2.0 * (r_range[1] - r_range[0]) + r_range[0]
        pose[:, 1] = (pose[:, 1] + 1.0) / 2.0 * (theta_range[1] - theta_range[0]) + theta_range[0]
        pose[:, 2] = (pose[:, 2] + 1.0) / 2.0 * (psi_range[1] - psi_range[0]) + psi_range[0]
    else:
        raise Exception('Error in data format of V shape: {}'.format(pose.shape))
    return pose


def read_images(data_dir, res, max_size=None):
    print('Going to read image file list')
    files_list = glob.glob(os.path.abspath(os.path.join(data_dir, 'images', '*.jpg')))
    print('Done. Starting sorting.')
    # files_list.sort()  # make sure we're reading the images in order later
    files_list = natsorted(files_list)
    print('Done. Before images_np init')
    if max_size is not None:
        size_data = max_size
    else:
        size_data = len(files_list)
    images_np = np.zeros((size_data, res, res, 3)).astype(np.float32)
    print('Done. Going to read images.')
    idx = 0
    for img_name in files_list:
        # read in image with cv2 (pixel order BGR)
        im = cv2.imread(img_name, cv2.IMREAD_COLOR)

        if im.shape[0] is not res or im.shape[1] is not res:
            im = cv2.resize(im, (res, res))

        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        if idx % 10000 == 0:
            print('image idx = {}'.format(idx))
        idx = idx + 1
        if idx == size_data:
            # reached the last point -- exit loop of images
            break

    print('Done reading {} images.'.format(images_np.shape[0]))
    return images_np


def create_dataset_csv(data_dir, batch_size, res, max_size=None):
    print('Going to read file list')
    files_list = glob.glob(os.path.join(data_dir, 'images' + os.sep + '*.jpg'))  # took out the preceding images dir
    print('Done. Starting sorting.')
    # files_list.sort()  # make sure we're reading the images in order later
    files_list = natsorted(files_list)

    print('Done. Before images_np init')
    if max_size is not None:
        size_data = max_size
    else:
        size_data = len(files_list)
    images_np = np.zeros((size_data, res, res, 3)).astype(np.float32)

    print('Going to read csv file.')
    # prepare state R THETA PSI as np array reading from a file
    raw_table = np.loadtxt(os.path.abspath(os.path.join(data_dir, 'state_data.csv')), delimiter=',')  # changed name of csv and delimiter from space to comma
    raw_table = raw_table[:size_data, :]

    print('Done. Going to read images.')

    idx = 0
    for file in files_list:
        # read in image with cv2 (pixel order BGR)
        im = cv2.imread(file, cv2.IMREAD_COLOR)

        if im.shape[0] is not res or im.shape[1] is not res:
            im = cv2.resize(im, (res, res))

        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        if idx % 10000 == 0:
            print('image idx = {}'.format(idx))
        idx = idx + 1
        if idx == size_data:
            # reached the last point -- exit loop of images
            break

    # sanity check
    if raw_table.shape[0] != images_np.shape[0]:
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(images_np.shape[0], raw_table.shape[0]))
    raw_table.astype(np.float32)

    # print some useful statistics
    print("Average state values: {}".format(np.mean(raw_table, axis=0)))
    print("Median  state values: {}".format(np.median(raw_table, axis=0)))
    print("STD of  state values: {}".format(np.std(raw_table, axis=0)))
    print("Max of  state values: {}".format(np.max(raw_table, axis=0)))
    print("Min of  state values: {}".format(np.min(raw_table, axis=0)))

    # normalize state variables to [-1, 1] range
    raw_table = normalize_state(raw_table)

    img_train, img_test, state_train, state_test = train_test_split(images_np, raw_table, test_size=0.1, random_state=42)

    # calculate number of batches
    num_train_imgs = img_train.shape[0]
    n_batches_train = (num_train_imgs + batch_size - 1) // batch_size
    num_test_imgs = img_test.shape[0]
    n_batches_test = (num_test_imgs + batch_size - 1) // batch_size
    print("Amount of training data: {}".format(num_train_imgs))
    print("Number of training batches: {}".format(n_batches_train))
    print("Amount of test data: {}".format(num_test_imgs))
    print("Number of test batches: {}".format(n_batches_test))

    # combine image and state data
    ds_train = [img_train, state_train]
    ds_test = [img_test, state_test]

    return ds_train, ds_test, n_batches_train, n_batches_test


def create_dataset_filepaths(data_dir, batch_size, res, max_size=None):
    print('Going to read file list')
    files_list = glob.glob(os.path.abspath(os.path.join(data_dir, 'images', '*.jpg')))  # took out the preceding images dir
    print('Done. Starting sorting.')
    # files_list.sort()  # make sure we're reading the images in order later
    files_list = natsorted(files_list)
    print('Done. Calculating data size and capping at max_size if using')
    if max_size is not None:
        size_data = max_size
    else:
        size_data = len(files_list)

    print('Going to read csv file.')
    # prepare state R THETA PSI as np array reading from a file
    raw_table = np.loadtxt(data_dir + os.sep + 'state_data.csv', delimiter=',')  # changed name of csv and delimiter from space to comma
    raw_table = raw_table[:size_data, :]

    # sanity check
    if raw_table.shape[0] != len(files_list):
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(len(files_list), raw_table.shape[0]))
    raw_table.astype(np.float32)

    # print some useful statistics
    print("Average state values: {}".format(np.mean(raw_table, axis=0)))
    print("Median  state values: {}".format(np.median(raw_table, axis=0)))
    print("STD of  state values: {}".format(np.std(raw_table, axis=0)))
    print("Max of  state values: {}".format(np.max(raw_table, axis=0)))
    print("Min of  state values: {}".format(np.min(raw_table, axis=0)))

    # normalize state variables to [-1, 1] range
    raw_table = normalize_state(raw_table)

    img_train, img_test, state_train, state_test = train_test_split(files_list, raw_table, test_size=0.1, random_state=42)

    # calculate number of batches
    num_train_imgs = len(img_train)
    n_batches_train = (num_train_imgs + batch_size - 1) // batch_size
    num_test_imgs = len(img_test)
    n_batches_test = (num_test_imgs + batch_size - 1) // batch_size
    print("Amount of training data: {}".format(num_train_imgs))
    print("Number of training batches: {}".format(n_batches_train))
    print("Amount of test data: {}".format(num_test_imgs))
    print("Number of test batches: {}".format(n_batches_test))

    # combine image and state data
    ds_train = [img_train, state_train]
    ds_test = [img_test, state_test]

    return ds_train, ds_test, n_batches_train, n_batches_test


def create_unsup_dataset_multiple_sources(data_dir_list, batch_size, res):
    # load all the images in one single large dataset
    images_np = np.empty((0, res, res, 3)).astype(np.float32)
    for data_dir in data_dir_list:
        img_array = read_images(data_dir, res, max_size=None)
        images_np = np.concatenate((images_np, img_array), axis=0)
    # make fake distances to target as -1
    num_items = images_np.shape[0]
    print('Real_life dataset has {} images total'.format(num_items))
    raw_table = (-1.0 * np.ones((num_items, 3))).astype(np.float32)
    # separate the actual dataset:
    img_train, img_test, state_train, state_test = train_test_split(images_np, raw_table, test_size=0.1, random_state=42)
    # calculate number of batches
    num_train_imgs = img_train.shape[0]
    n_batches_train = (num_train_imgs + batch_size - 1) // batch_size
    num_test_imgs = img_test.shape[0]
    n_batches_test = (num_test_imgs + batch_size - 1) // batch_size
    print("Amount of training data: {}".format(num_train_imgs))
    print("Number of training batches: {}".format(n_batches_train))
    print("Amount of test data: {}".format(num_test_imgs))
    print("Number of test batches: {}".format(n_batches_test))

    # combine image and state data
    ds_train = [img_train, state_train]
    ds_test = [img_test, state_test]

    return ds_train, ds_test, n_batches_train, n_batches_test


def create_test_dataset_csv(data_dir, res, read_table=True):
    # prepare image dataset from a folder
    print('Going to read file list')
    files_list = glob.glob(os.path.abspath(os.path.join(data_dir, 'images', '*.jpg')))  # took out the preceding images dir
    print('Done. Starting sorting.')
    # files_list.sort()  # make sure we're reading the images in order later
    files_list = natsorted(files_list)
    print('Done. Before images_np init')
    images_np = np.zeros((len(files_list), res, res, 3)).astype(np.float32)
    print('After images_np init')
    idx = 0
    for file in files_list:
        # read in image with cv2 (pixel order BGR)
        im = cv2.imread(file, cv2.IMREAD_COLOR)

        if im.shape[0] is not res or im.shape[1] is not res:
            im = cv2.resize(im, (res, res))

        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        idx = idx + 1

    if not read_table:
        return images_np, None

    # prepare state R THETA PSI as np array reading from a file
    raw_table = np.loadtxt(os.path.abspath(os.path.join(data_dir, 'state_data.csv')), delimiter=',')  # changed name of csv file and delimiter from space to comma
    # sanity check
    if raw_table.shape[0] != images_np.shape[0]:
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(images_np.shape[0], raw_table.shape[0]))
    raw_table.astype(np.float32)

    # print some useful statistics
    print("Average state values: {}".format(np.mean(raw_table, axis=0)))
    print("Median  state values: {}".format(np.median(raw_table, axis=0)))
    print("STD of  state values: {}".format(np.std(raw_table, axis=0)))
    print("Max of  state values: {}".format(np.max(raw_table, axis=0)))
    print("Min of  state values: {}".format(np.min(raw_table, axis=0)))

    return images_np, raw_table
