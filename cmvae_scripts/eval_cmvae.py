import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

import cmvae_models.cmvae
import cmvae_utils

cmvae_utils.dataset_utils.seed_environment()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Directory where the images/state data is contained', default="", type=str)
parser.add_argument('--interpolation_dir', help='Directory where the images/state data is contained for interpolation', default="", type=str)
parser.add_argument('--weights_path', help='Directory where the pretrained model is', default="", type=str)
args = parser.parse_args()

if args.data_dir == '':
    print('No data directory specified, quitting!')
    quit()

# define training meta parameters
data_dir = args.data_dir
weights_path = args.weights_path
output_dir = os.path.join(data_dir, 'results')
interpolation_dir = args.interpolation_dir
output_dir_interp = os.path.join(interpolation_dir, 'results')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(output_dir_interp):
    os.makedirs(output_dir_interp)

# DEFINE TESTING META PARAMETERS
n_z = 10
img_res = 64
num_imgs_display = 8
columns = 4
rows = 4

num_interp_z = 9
z_range_mural = [-0.02, 0.02]
z_num_mural = 11

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load test dataset
images_np, raw_table = cmvae_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res)
print('Done with dataset')

images_np = images_np[:1000, :]
raw_table = raw_table[:1000, :]

# create model
model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=3, res=img_res)

print('Loading weights from {}'.format(os.path.join(output_dir, weights_path)))
model.load_weights(weights_path)

img_recon, gate_recon, means, stddev, z = model(images_np, mode=0)
img_recon = img_recon.numpy()
gate_recon = gate_recon.numpy()
z = z.numpy()

# de-normalization of gates and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon)

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
fig.savefig(os.path.join(output_dir, 'reconstruction_results.png'))

images_np_interp, raw_table_interp = cmvae_utils.dataset_utils.create_test_dataset_csv(interpolation_dir, img_res)

img_recon_interps, gate_recon_interps, means_interps, stddev_interps, z_interps = model(images_np_interp, mode=0)
img_recon_interps = img_recon_interps.numpy()
gate_recon_interps = gate_recon_interps.numpy()
z_interps = z_interps.numpy()

# de-normalization of gates and images
images_np_interp = ((images_np_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon_interps = ((img_recon_interps + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon_interps = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon_interps)

# show interpolation btw two images in latent space
z_r_min = z_interps[0, :]
z_r_max = z_interps[1, :]
z_r_interp = cmvae_utils.geom_utils.interp_vector(z_r_min, z_r_max, num_interp_z)

z_theta_min = z_interps[2, :]
z_theta_max = z_interps[3, :]
z_theta_interp = cmvae_utils.geom_utils.interp_vector(z_theta_min, z_theta_max, num_interp_z)

z_min_psi = z_interps[4, :]
z_max_psi = z_interps[5, :]
z_psi_interp = cmvae_utils.geom_utils.interp_vector(z_min_psi, z_max_psi, num_interp_z)

z_interp = [z_r_interp, z_theta_interp, z_psi_interp]

idx = 0
for z_int in z_interp:
    # get the image predictions
    img_recon_interp, gate_recon_interp = model.decode(z_int, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    gate_recon_interp = gate_recon_interp.numpy()

    # de-normalization of gates and images
    img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
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

    fig.savefig(os.path.join(output_dir_interp, 'state_stats_interpolation_results_{}.png'.format(label)))

    # plot the interpolated images
    fig2 = plt.figure(figsize=(96, 96))
    columns = num_interp_z + 2
    rows = 1
    fig2.add_subplot(rows, columns, 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np_interp[(2 * idx), :])
    plt.axis('off')
    plt.imshow(img_display)
    for i in range(1, num_interp_z + 1):
        fig2.add_subplot(rows, columns, i + 1)
        img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
        plt.axis('off')
        plt.imshow(img_display)
    fig2.add_subplot(rows, columns, num_interp_z + 2)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np_interp[(2 * idx) + 1, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig2.savefig(os.path.join(output_dir_interp, 'reconstruction_interpolation_results_{}.png'.format(label)))
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
    img_recon_interp, gate_recon_interp = model.decode(z, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
    plt.axis('off')
    plt.imshow(img_display)
fig3.savefig(os.path.join(output_dir, 'z_mural.png'))

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

del model
