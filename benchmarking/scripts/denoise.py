import argparse
import os
import shutil

import cv2
import yaml

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_global_config.yml'), 'r') as f:
    cmvae_global_config = yaml.load(f, Loader=yaml.UnsafeLoader)

import tensorflow as tf

if cmvae_global_config['use_cpu_only']:
    tf.config.set_visible_devices([], 'GPU')

import csv

import numpy as np
from matplotlib import pyplot as plt

import cmvae_utils.dataset_utils
import cmvae_utils.stats_utils
import cmvae_utils.geom_utils
from gym_underwater.utils import load_environment_config, load_cmvae_inference_config, output_devices, count_directories_in_directory, parse_command_args, \
    tensorflow_seeding, duplicate_directory, load_cmvae, output_command_line_arguments

env_config = load_environment_config(project_dir)
cmvae_inference_config = load_cmvae_inference_config(project_dir)

# parse_command_args(env_config, cmvae_inference_config)

tensorflow_seeding(env_config['seed'])

# assert (cmvae_inference_config['test_dir'] != ''), 'No data directory specified, quitting'
assert (cmvae_inference_config['weights_path'] != ''), 'Require pre-trained weights'
# assert (cmvae_inference_config['interpolation_dir'] != ''), 'Require interpolation directory'

parser = argparse.ArgumentParser()
parser.add_argument('--dirs', help='Directories (comma separated e.g. <dir1,dir2,dir3>', default=None, type=str)
parser.add_argument('--dir_output', help='Output Directory', default=None, type=str)
args = parser.parse_args()

dirs = args.dirs.split(',')

# interpolation_dir = cmvae_inference_config['interpolation_dir']

cmvae = load_cmvae(cmvae_global_config=cmvae_global_config, weights_path=cmvae_inference_config['weights_path'])

output_dir = args.dir_output

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

# DEFINE TESTING META PARAMETERS
num_imgs_display = 1
columns = 2
rows = 2 * num_imgs_display
# n_z = cmvae_global_config['n_z']
img_res = cmvae.img_res

# num_interp_z = 9
# z_range_mural = [-0.02, 0.02]
# z_num_mural = 11

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
imgs_np = []
imgs_recon = []
gates_np = []
gates_pred = []
zs = []

# Load test dataset
for dir in dirs:
    images_np, raw_table = cmvae_utils.dataset_utils.create_test_dataset_csv(dir, img_res)
    print('Done with dataset')

    images_np = images_np[:1000, :]
    raw_table = raw_table[:1000, :]

    img_recon, gate_recon, means, stddev, z = cmvae(images_np, training=False, mode=0)
    img_recon = img_recon.numpy()
    gate_recon = gate_recon.numpy()
    z = z.numpy()

    # de-normalization of gates and images
    images_np = cmvae_utils.dataset_utils.denormalize_image(images_np)
    img_recon = cmvae_utils.dataset_utils.denormalize_image(img_recon)
    gate_recon = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon)

    output_dir_denoise = os.path.join(dir, 'images_recon')

    if os.path.exists(output_dir_denoise):
        shutil.rmtree(output_dir_denoise)

    os.makedirs(output_dir_denoise)

    for i, img in enumerate(img_recon):
        cv2.imwrite(os.path.join(output_dir_denoise, f'{i}.jpg'), img)

    imgs_np.append(images_np)
    imgs_recon.append(img_recon)

    gates_np.append(raw_table)
    gates_pred.append(gate_recon)

    zs.append(z)

filename_img_output = os.path.join(output_dir, 'prediction_img.csv')

with open(filename_img_output, 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['MAE', 'Standard Error', 'Max Error'])

cmvae_utils.stats_utils.calculate_img_stats(imgs_recon[0].astype(np.int32), imgs_recon[1].astype(np.int32), filename_img_output)
cmvae_utils.stats_utils.calculate_gate_stats(gates_pred[0], gates_pred[1], output_dir)
cmvae_utils.stats_utils.calculate_z_stats(zs[0], zs[1], output_dir)

# show some reconstruction figures
fig = plt.figure(figsize=(20, 20))
for i in range(1, num_imgs_display + 2):
    idx_orig = (i - 1) * 2 + 1
    fig.add_subplot(rows, columns, idx_orig)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(imgs_np[i - 1][0, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig.add_subplot(rows, columns, idx_orig + 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(imgs_recon[i - 1][0, :])
    plt.axis('off')
    plt.imshow(img_display)
fig.savefig(os.path.join(output_dir, 'noiseless_noisy_recon.pdf'))
