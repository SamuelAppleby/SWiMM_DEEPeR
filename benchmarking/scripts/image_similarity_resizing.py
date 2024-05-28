import argparse
import csv
import os

import numpy as np

from cmvae_utils.dataset_utils import load_img_from_file_or_array_and_resize_cv2
from cmvae_utils.stats_utils import calculate_img_stats

parser = argparse.ArgumentParser()
parser.add_argument('--dirs_orig', help='Original directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dirs_scaled', help='Scaled directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dir_output', help='Output Directory', default=None, type=str)
args = parser.parse_args()

dirs_orig = args.dirs_orig.split(',')
dirs_scaled = args.dirs_scaled.split(',')

res_scaled = (64, 64, 3)

assert isinstance(dirs_orig, list) and isinstance(dirs_scaled, list), 'Invalid Directories'
assert args.dir_output is not None, 'Invalid output directory'

if not os.path.exists(args.dir_output):
    os.makedirs(args.dir_output)

with open(os.path.join(args.dir_output, 'resizing.csv'), 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['Original Directory', 'Scaled Directory', 'MAE', 'Standard Error', 'Max Error'])

    for dir_orig, dir_scaled in zip(dirs_orig, dirs_scaled):
        writer.writerow([dir_orig, dir_scaled])

for dir_orig, dir_scaled in zip(dirs_orig, dirs_scaled):
    unity_scaled = np.zeros(np.concatenate(([len(os.listdir(dir_scaled))], res_scaled))).astype(np.int8)
    python_scaled = np.zeros(np.concatenate(([len(os.listdir(dir_orig))], res_scaled))).astype(np.int8)

    for idx, filename in enumerate(os.listdir(dir_scaled)):
        unity_scaled[idx, :] = load_img_from_file_or_array_and_resize_cv2(file=os.path.join(dir_scaled, filename), res=res_scaled, normalise=False)

    for idx, filename in enumerate(os.listdir(dir_orig)):
        python_scaled[idx, :] = load_img_from_file_or_array_and_resize_cv2(file=os.path.join(dir_orig, filename), res=res_scaled, normalise=False)

    calculate_img_stats(unity_scaled, python_scaled, os.path.join(args.dir_output, 'resizing.csv'))
