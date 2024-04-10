import argparse
import csv
import os

import numpy as np

from gym_underwater.utils.utils import read_files_from_dir, image_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--dirs_orig', help='Original directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dirs_scaled', help='Scaled directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dir_output', help='Output Directory', default=None, type=str)
args = parser.parse_args()

dirs_orig = args.dirs_orig.split(',')
dirs_scaled = args.dirs_scaled.split(',')

assert isinstance(dirs_orig, list) and isinstance(dirs_scaled, list), 'Invalid Directories'
assert args.dir_output is not None, 'Invalid output directory'

if not os.path.exists(args.dir_output):
    os.makedirs(args.dir_output)

with open(os.path.join(args.dir_output, 'resizing.csv'), 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(['Original Directory', 'Scaled Directory', 'Mean', 'Standard Deviation'])

    for dir_orig, dir_scaled in zip(dirs_orig, dirs_scaled):
        unity_scaled = read_files_from_dir([os.path.join(dir_scaled, filename) for filename in os.listdir(dir_scaled)], resize_img=False)
        python_scaled = read_files_from_dir([os.path.join(dir_orig, filename) for filename in os.listdir(dir_orig)], resize_img=True)

        sims = []

        for image_1, image_2 in zip(unity_scaled, python_scaled):
            sims.append(image_similarity(image_1, image_2))

        writer.writerow([dir_orig, dir_scaled, np.average(sims), np.std(sims)])
