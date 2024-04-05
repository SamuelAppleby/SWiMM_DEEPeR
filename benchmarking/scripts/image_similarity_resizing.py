import argparse
import csv
import os

import numpy as np
from PIL import Image
from numpy.random import default_rng

from gym_underwater.utils.utils import read_files_from_dir, image_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--dir_orig', help='Directory 1', default=None, type=str)
parser.add_argument('--dir_unity_scaled', help='Directory 2', default=None, type=str)
parser.add_argument('--num_samples', help='Number of samples 2', default=None, type=int)
args = parser.parse_args()

assert args.dir_orig is not None and args.dir_unity_scaled is not None and args.num_samples is not None, 'Invalid arguments'

file_names_orig = os.listdir(args.dir_orig)
file_names_unity = os.listdir(args.dir_unity_scaled)

assert args.num_samples <= len(file_names_orig), 'Asking for more image samples than are available'
numbers = default_rng().choice(len(file_names_orig), size=args.num_samples, replace=False)

orig_files_rand = []
unity_files_rand = []

for num in numbers:
    orig_files_rand.append(os.path.join(args.dir_orig, file_names_orig[num]))
    unity_files_rand.append(os.path.join(args.dir_unity_scaled, file_names_unity[num]))

unity_scaled = read_files_from_dir(unity_files_rand, resize_img=False)
python_scaled = read_files_from_dir(orig_files_rand, resize_img=True)

pair_res = np.empty(0)

for num in range(len(unity_scaled)):
    sim = image_similarity(unity_scaled[num], python_scaled[num])
    pair_res = np.append(pair_res, sim)

output_dir = os.path.join(args.dir_orig, os.pardir, os.pardir, 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'resizing.csv'), 'a', newline='', encoding='UTF8') as f:
    orig_res = Image.open(os.path.join(args.dir_orig, '1.jpg'))

    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(['Original Width', 'Original Height', 'Scaled Width', 'Scaled Height', 'Average', 'Standard Deviation'])

    writer.writerow([orig_res.size[0], orig_res.size[1], unity_scaled.shape[1], unity_scaled.shape[2], np.average(pair_res), np.std(pair_res)])

print(str(args.num_samples) + ' samples between unity and python\n' + 'Average:' + str(np.average(pair_res)) + '\n' + 'Standard Deviation:' + str(np.std(pair_res)))
