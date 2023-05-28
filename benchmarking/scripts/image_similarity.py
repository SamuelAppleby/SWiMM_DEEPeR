import argparse
import csv
import os
from random import random

from PIL import Image
import numpy as np
from skimage.transform import resize
from numpy.linalg import norm
from numpy.random import default_rng


def image_similarity(image_1, image_2):
    image_1 = image_1.flatten()
    image_2 = image_2.flatten()

    test1 = np.dot(image_1, image_2) / (norm(image_1) * norm(image_2))

    image_1 = image_1 / 255
    image_2 = image_2 / 255

    test2 = np.dot(image_1, image_2) / (norm(image_1) * norm(image_2))

    return np.dot(image_1, image_2) / (norm(image_1) * norm(image_2))


def read_files_from_dir(dir, resize_img=False):
    arr = []

    for file in dir:
        if file.endswith('.jpg'):
            img = np.array(Image.open(file))
            if resize_img:
                arr.append(resize(img, (64, 64), 0, preserve_range=True).astype(np.uint8))
            else:
                arr.append(img)

    return arr


def unity_python_similarity(dir_orig, dir_unity_sample, num_samples, output_dir):
    orig_res = Image.open(os.path.join(dir_orig, '1.jpg'))
    file_names_orig = os.listdir(dir_orig)
    file_names_unity = os.listdir(dir_unity_sample)

    rng = default_rng()
    numbers = rng.choice(len(file_names_orig), size=num_samples, replace=False)

    orig_files_rand = []
    unity_files_rand = []

    for num in numbers:
        orig_files_rand.append(os.path.join(dir_orig, file_names_orig[num]))
        unity_files_rand.append(os.path.join(dir_unity_sample, file_names_unity[num]))

    python_scaled = read_files_from_dir(orig_files_rand, resize_img=True)
    unity_scaled = read_files_from_dir(unity_files_rand, resize_img=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, 'resizing.csv')

    with open(output_dir, 'a', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(['full_size_dir', 'unity_resize_dir', 'width', 'height', 'cosine_similarity'])

        pair_res = []

        for num in range(len(unity_scaled)):
            sim = image_similarity(unity_scaled[num], python_scaled[num])
            pair_res.append(sim)
            writer.writerow(
                [orig_files_rand[num], unity_files_rand[num], orig_res.size[0], orig_res.size[1], sim])

        pair_res = np.asarray(pair_res)
        print(str(num_samples) + ' samples between unity and python\n' + 'Average:' + str(np.average(pair_res)) + '\n' + 'Standard Deviation:' + str(np.std(pair_res)))


def cross_resolution_similarity(dir_low_scaled, dir_high_scaled, dir_raw, num_samples, output_dir):
    file_names_low = os.listdir(dir_low_scaled)
    file_names_high = os.listdir(dir_high_scaled)
    file_names_raw = os.listdir(dir_raw)

    rng = default_rng()
    numbers = rng.choice(len(file_names_raw), size=num_samples, replace=False)

    files_rand_low = []
    files_rand_high = []
    files_rand_raw = []

    for num in numbers:
        files_rand_low.append(os.path.join(dir_low_scaled, file_names_low[num]))
        files_rand_high.append(os.path.join(dir_high_scaled, file_names_high[num]))
        files_rand_raw.append(os.path.join(dir_raw, file_names_raw[num]))

    low_scaled = read_files_from_dir(files_rand_low, resize_img=False)
    high_scaled = read_files_from_dir(files_rand_high, resize_img=False)
    raw = read_files_from_dir(files_rand_raw, resize_img=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, 'cross_resolution.csv')

    with open(output_dir, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(['image', 'low_high_sim', 'low_raw_sim', 'high_raw_sim'])

        res_dict = {
            dir_low_scaled + '&' + dir_high_scaled: [],
            dir_low_scaled + '&' + dir_raw: [],
            dir_high_scaled + '&' + dir_raw: []
        }

        for num in range(len(raw)):
            res_dict[dir_low_scaled + '&' + dir_high_scaled].append(image_similarity(low_scaled[num], high_scaled[num]))
            res_dict[dir_low_scaled + '&' + dir_raw].append(image_similarity(low_scaled[num], raw[num]))
            res_dict[dir_high_scaled + '&' + dir_raw].append(image_similarity(high_scaled[num], raw[num]))

            writer.writerow(
                [num, res_dict[dir_low_scaled + '&' + dir_high_scaled][num], res_dict[dir_low_scaled + '&' + dir_raw][num], res_dict[dir_high_scaled + '&' + dir_raw][num]])

        res_dict[dir_low_scaled + '&' + dir_high_scaled] = np.asarray(res_dict[dir_low_scaled + '&' + dir_high_scaled])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_orig', help='Directory 1', default=None, type=str)
    parser.add_argument('--dir_unity_scaled', help='Directory 2', default=None, type=str)

    parser.add_argument('--dir_high_scaled', help='Directory 2', default=None, type=str)
    parser.add_argument('--dir_low_scaled', help='Directory 2', default=None, type=str)
    parser.add_argument('--dir_raw', help='Directory 2', default=None, type=str)

    parser.add_argument('--dir_output', help='Directory 2', default=None, type=str)
    parser.add_argument('--num_samples', help='Directory 2', default=1000, type=int)

    parser.add_argument('--resize', help='Run experiment  for unity/python', action='store_true')
    parser.add_argument('--cross_resolution', help='Run experiment  for cross resolution', action='store_true')
    args = parser.parse_args()

    if args.resize:
        unity_python_similarity(args.dir_orig, args.dir_unity_scaled, args.num_samples, args.dir_output)
    elif args.cross_resolution:
        cross_resolution_similarity(args.dir_low_scaled, args.dir_high_scaled, args.dir_raw, args.num_samples, args.dir_output)


if __name__ == '__main__':
    main()
