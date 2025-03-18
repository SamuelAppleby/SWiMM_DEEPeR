import argparse
import csv
import os

import numpy as np

from cmvae_utils.dataset_utils import load_img_from_file_or_array_and_resize_cv2
from cmvae_utils.stats_utils import calculate_img_stats
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', help='Directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dir_output', help='Output Directory', default=None, type=str)
args = parser.parse_args()

dirs = args.dirs.split(',')
file_output_name = 'results.csv'
res_scaled = (64, 64, 3)

assert isinstance(dirs, list), 'Invalid directory'
assert args.dir_output is not None, 'Invalid output directory'

if not os.path.exists(args.dir_output):
    os.makedirs(args.dir_output)

dirs_names_images = {}
for directory in dirs:
    dirs_names_images.update({
        directory: np.zeros(np.concatenate(([len(os.listdir(directory))], res_scaled))).astype(np.int8)
    })

    for idx, filename in enumerate(os.listdir(directory)):
        dirs_names_images[directory][idx, :] = load_img_from_file_or_array_and_resize_cv2(file=os.path.join(directory, filename), res=res_scaled, normalise=False)

combinations = list(combinations(list(dirs_names_images.keys()), 2))

with open(os.path.join(args.dir_output, file_output_name), 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['Directory 1', 'Directory 2', 'MAE', 'Standard Error', 'Max Error'])

    for combination in combinations:
        writer.writerow([combination[0], combination[1]])

for combination in combinations:
    calculate_img_stats(dirs_names_images[combination[0]], dirs_names_images[combination[1]], os.path.join(args.dir_output, file_output_name))

