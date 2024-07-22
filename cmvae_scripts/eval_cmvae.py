import os
import yaml

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from gym_underwater.utils.utils import load_environment_config, load_cmvae_inference_config, output_devices, count_directories_in_directory, parse_command_args, \
    tensorflow_seeding, duplicate_directory, load_cmvae

env_config = load_environment_config(project_dir)
cmvae_inference_config = load_cmvae_inference_config(project_dir)

parse_command_args(env_config, cmvae_inference_config)

tensorflow_seeding(env_config['seed'])

assert (cmvae_inference_config['test_dir'] != ''), 'No data directory specified, quitting'
assert (cmvae_inference_config['weights_path'] != ''), 'Require pre-trained weights'
assert (cmvae_inference_config['interpolation_dir'] != ''), 'Require interpolation directory'

test_dir = cmvae_inference_config['test_dir']
interpolation_dir = cmvae_inference_config['interpolation_dir']

cmvae = load_cmvae(cmvae_global_config=cmvae_global_config, weights_path=cmvae_inference_config['weights_path'])

output_dir = os.path.join(os.path.dirname(cmvae_inference_config['weights_path']), 'inference_results')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = os.path.join(output_dir, str(count_directories_in_directory(output_dir)))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DEFINE TESTING META PARAMETERS
num_imgs_display = 8
columns = 4
rows = 4
n_z = cmvae_global_config['n_z']
img_res = cmvae.img_res

num_interp_z = 9
z_range_mural = [-0.02, 0.02]
z_num_mural = 11

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# Load test dataset
images_np, raw_table = cmvae_utils.dataset_utils.create_test_dataset_csv(test_dir, img_res)
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

with open(os.path.join(output_dir, 'prediction_img.csv'), 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['MAE', 'Standard Error', 'Max Error'])

cmvae_utils.stats_utils.calculate_img_stats(img_recon.astype(np.int32), images_np.astype(np.int32), os.path.join(output_dir, 'prediction_img.csv'))

# get stats for gate reconstruction
cmvae_utils.stats_utils.calculate_gate_stats(gate_recon, raw_table, output_dir)

# show some reconstruction figures
fig = plt.figure(figsize=(20, 20))
for i in range(1, num_imgs_display + 1):
    idx_orig = (i - 1) * 2 + 1
    fig.add_subplot(rows, columns, idx_orig)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[i - 1, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig.add_subplot(rows, columns, idx_orig + 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon[i - 1, :])
    plt.axis('off')
    plt.imshow(img_display)
fig.savefig(os.path.join(output_dir, 'reconstruction_results.pdf'))

images_np_interps, raw_table_interps = cmvae_utils.dataset_utils.create_test_dataset_csv(interpolation_dir, img_res)

img_recon_interps, gate_recon_interps, means_interps, stddev_interps, z_interps = cmvae(images_np_interps, mode=0)
img_recon_interps = img_recon_interps.numpy()
gate_recon_interps = gate_recon_interps.numpy()
z_interps = z_interps.numpy()

# de-normalization of gates and images
images_np_interps = cmvae_utils.dataset_utils.denormalize_image(images_np_interps)
img_recon_interps = cmvae_utils.dataset_utils.denormalize_image(img_recon_interps)
gate_recon_interps = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon_interps)

# show interpolation btw two images in latent space
z_r_interp = cmvae_utils.geom_utils.interp_vector(z_interps[0, :], z_interps[1, :], num_interp_z)
z_theta_interp = cmvae_utils.geom_utils.interp_vector(z_interps[2, :], z_interps[3, :], num_interp_z)
z_psi_interp = cmvae_utils.geom_utils.interp_vector(z_interps[4, :], z_interps[5, :], num_interp_z)

z_interp = [z_r_interp, z_theta_interp, z_psi_interp]

idx = 0
for z_int in z_interp:
    # get the image predictions
    img_recon_interp, gate_recon_interp = cmvae.decode(z_int, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    gate_recon_interp = gate_recon_interp.numpy()

    # de-normalization of gates and images
    img_recon_interp = cmvae_utils.dataset_utils.denormalize_image(img_recon_interp)
    gate_recon_interp = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon_interp)

    # join predictions with array and print
    indices = np.array([np.arange(num_interp_z)]).transpose()
    results = np.concatenate((indices, gate_recon_interp), axis=1)
    print('Img index | Predictions: = \n{}'.format(results))

    fig, axs = plt.subplots(1, 3, tight_layout=True)
    axs[0].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 0], 'b-', label='r')
    axs[1].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 1], 'b-', label=r'$\theta$')
    axs[2].plot(np.arange(gate_recon_interp.shape[0]), gate_recon_interp[:, 2], 'b-', label=r'$\psi$')

    for i in range(3):
        # axs[idx].grid()
        y_ticks_array = gate_recon_interp[:, i][np.array([0, gate_recon_interp[:, i].shape[0] - 1])]
        y_ticks_array = np.around(y_ticks_array, decimals=1)
        if i > 0:
            y_ticks_array = y_ticks_array
        axs[i].set_yticks(y_ticks_array)
        axs[i].set_xticks(np.array([0, 9]))
        axs[i].set_xticklabels((r'$I_a$', r'$I_b$'))

    axs[0].set_title(r'$r$')
    axs[1].set_title(r'$\theta$')
    axs[2].set_title(r'$\psi$')

    axs[0].set_ylabel('[meter]')
    axs[1].set_ylabel(r'[deg]')
    axs[2].set_ylabel(r'[deg]')

    label = 'r' if idx == 0 else 'theta' if idx == 1 else 'psi'

    fig.savefig(os.path.join(output_dir, 'state_stats_interpolation_results_{}.pdf'.format(label)))

    # plot the interpolated images
    fig2 = plt.figure(figsize=(96, 96))
    columns = num_interp_z + 2
    rows = 1
    fig2.add_subplot(rows, columns, 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np_interps[(2 * idx), :])
    plt.axis('off')
    plt.imshow(img_display)
    for i in range(1, num_interp_z + 1):
        fig2.add_subplot(rows, columns, i + 1)
        img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
        plt.axis('off')
        plt.imshow(img_display)
    fig2.add_subplot(rows, columns, num_interp_z + 2)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np_interps[(2 * idx) + 1, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig2.savefig(os.path.join(output_dir, 'reconstruction_interpolation_results_{}.pdf'.format(label)))
    idx += 1

# new plot traveling through latent space
fig3 = plt.figure(figsize=(96, 96))
columns = z_num_mural
rows = n_z
z_values = cmvae_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
for i in range(1, z_num_mural * n_z + 1):
    fig3.add_subplot(rows, columns, i)
    z = np.zeros((1, n_z)).astype(np.float32)
    z[0, int((i - 1) / columns)] = z_values[i % columns - 1]
    img_recon_interp, gate_recon_interp = cmvae.decode(z, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    gate_recon_interp = gate_recon_interp.numpy()

    img_recon_interp = cmvae_utils.dataset_utils.denormalize_image(img_recon_interp[0, :])

    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
    plt.axis('off')
    plt.imshow(img_display)
fig3.savefig(os.path.join(output_dir, 'z_mural.pdf'))

# single-channel version of above
# fig4 = plt.figure(figsize=(96, 96))
# columns = z_num_mural
# rows = 1
# z_values = cmvae_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
# for i in range(1, z_num_mural + 1):
#     fig4.add_subplot(rows, columns, i)
#     z = np.zeros((1, n_z)).astype(np.float32)
#     z[0, 2] = z_values[i - 1]  # interp across yaw feature
#     z[0, 0] = 0.02  # but hard code distance feature to 'close'
#     img_recon_interp, state_recon_interp = model.decode(z, mode=0)
#     img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
#     img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
#     plt.axis('off')
#     plt.imshow(img_display)
# fig4.savefig(os.path.join(output_dir, 'yaw_up_close.png'))

config_dir = os.path.join(output_dir, 'configs')

duplicate_directory(os.path.join(project_dir, 'configs'), config_dir, dirs_to_exclude=['hyperparams'], files_to_exclude=['cmvae_training_config.yml', 'callbacks.yml', 'env_wrapper.yml', 'server_config.json'])

output_devices(config_dir, tensorflow_device=True)
