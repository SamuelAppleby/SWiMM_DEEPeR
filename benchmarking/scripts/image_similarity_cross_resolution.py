import argparse
import csv
import os

import numpy as np
from numpy.random import default_rng

from gym_underwater.utils.utils import read_files_from_dir, image_similarity

RES_HIGH = (1920, 1080)
RES_LOW = (640, 360)

parser = argparse.ArgumentParser()
parser.add_argument('--dir_high_scaled', help='Directory 2', default=None, type=str)
parser.add_argument('--dir_low_scaled', help='Directory 2', default=None, type=str)
parser.add_argument('--dir_raw', help='Directory 2', default=None, type=str)
parser.add_argument('--num_samples', help='Directory 2', default=None, type=int)
args = parser.parse_args()

assert args.dir_high_scaled is not None and args.dir_low_scaled is not None and args.dir_raw is not None and args.num_samples is not None, 'Invalid arguments'

file_names_high = os.listdir(args.dir_high_scaled)
file_names_low = os.listdir(args.dir_low_scaled)
file_names_raw = os.listdir(args.dir_raw)

assert args.num_samples <= len(file_names_low), 'Asking for more image samples than are available'

rng = default_rng()
numbers = rng.choice(len(file_names_raw), size=args.num_samples, replace=False)

files_rand_high = []
files_rand_low = []
files_rand_raw = []

for num in numbers:
    files_rand_high.append(os.path.join(args.dir_high_scaled, file_names_high[num]))
    files_rand_low.append(os.path.join(args.dir_low_scaled, file_names_low[num]))
    files_rand_raw.append(os.path.join(args.dir_raw, file_names_raw[num]))

output_dir = os.path.join(args.dir_low_scaled, os.pardir, os.pardir, 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = os.path.join(output_dir, 'cross_resolution.csv')

with open(output_dir, 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)

    high_scaled = read_files_from_dir(files_rand_high, resize_img=False)
    low_scaled = read_files_from_dir(files_rand_low, resize_img=False)
    raw = read_files_from_dir(files_rand_raw, resize_img=False)

    res_dict = {
        (RES_HIGH[0], RES_HIGH[1], RES_LOW[0], RES_LOW[1]): np.empty(0),
        (RES_HIGH[0], RES_HIGH[1], raw[0].shape[0], raw[0].shape[1]): np.empty(0),
        (RES_LOW[0], RES_LOW[1], raw[0].shape[0], raw[0].shape[1]): np.empty(0)
    }

    for num in range(len(raw)):
        res_dict[(RES_HIGH[0], RES_HIGH[1], RES_LOW[0], RES_LOW[1])] = np.append(res_dict[(RES_HIGH[0], RES_HIGH[1], RES_LOW[0], RES_LOW[1])], image_similarity(high_scaled[num], low_scaled[num]))
        res_dict[(RES_HIGH[0], RES_HIGH[1], raw[0].shape[0], raw[0].shape[1])] = np.append(res_dict[(RES_HIGH[0], RES_HIGH[1], raw[0].shape[0], raw[0].shape[1])], image_similarity(high_scaled[num], raw[num]))
        res_dict[(RES_LOW[0], RES_LOW[1], raw[0].shape[0], raw[0].shape[1])] = np.append(res_dict[(RES_LOW[0], RES_LOW[1], raw[0].shape[0], raw[0].shape[1])], image_similarity(low_scaled[num], raw[num]))

    if f.tell() == 0:
        writer.writerow(['Original Width 1', 'Original Height 1', 'Original Width 2', 'Original Height 2', 'Mean Similarity', 'Standard Deviation'])

    for key, value in res_dict.items():
        writer.writerow([key[0], key[1], key[2], key[3], np.mean(value), np.std(value)])
