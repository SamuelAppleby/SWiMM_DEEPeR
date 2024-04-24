import argparse
import csv
import os

import numpy as np

from gym_underwater.utils.utils import read_files_from_dir, image_similarity
from itertools import combinations

RES_HIGH = (1920, 1080)
RES_LOW = (640, 360)
RES_RAW = (64, 64)

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', help='Directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dir_output', help='Output Directory', default=None, type=str)
args = parser.parse_args()

dirs = args.dirs.split(',')

assert isinstance(dirs, list), 'Invalid directory'
assert args.dir_output is not None, 'Invalid output directory'

dirs_names_images = {}
for directory in dirs:
    dirs_names_images.update({
        directory: read_files_from_dir([os.path.join(directory, filename) for filename in os.listdir(directory)], resize_img=False)
    })


combinations = list(combinations(list(dirs_names_images.keys()), 2))

if not os.path.exists(args.dir_output):
    os.makedirs(args.dir_output)

with open(os.path.join(args.dir_output, 'cross_resolution.csv'), 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(['Directory 1', 'Directory 2', 'Mean Similarity', 'Standard Deviation', 'Loss'])

    for combination in combinations:
        sims = []
        for image_1, image_2 in zip(dirs_names_images[combination[0]], dirs_names_images[combination[1]]):
            sims.append(image_similarity(image_1, image_2))

        sims_mean = np.mean(sims)
        writer.writerow([combination[0], combination[1], sims_mean, np.std(sims), 1 - sims_mean])
