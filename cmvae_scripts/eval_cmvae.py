import argparse
import os
import sys
import numpy as np
import tensorflow as tf

curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

import cmvae_models.cmvae
import cmvae_utils

cmvae_utils.dataset_utils.seed_environment()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Directory where the images/state data is contained', default="", type=str)
parser.add_argument('--weights_path', help='Directory where the pretrained model is', default="", type=str)
args = parser.parse_args()

if args.data_dir == '':
    print('No data directory specified, quitting!')
    quit()

# define training meta parameters
data_dir = args.data_dir
weights_path = args.weights_path
output_dir = os.path.join(os.path.dirname(weights_path), 'results')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# DEFINE TESTING META PARAMETERS
n_z = 10
img_res = 64
num_imgs_display = 8
columns = 4
rows = 4
read_table = True

num_interp_z = 10
idx_close = 615
idx_far = 703

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
images_np, raw_table = cmvae_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res, read_table=read_table)
print('Done with dataset')

images_np = images_np[:1000, :]
if read_table is True:
    raw_table = raw_table[:1000, :]

# create model
model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=3, res=img_res, trainable_model=True)

print('Loading weights from {}'.format(os.path.join(output_dir, weights_path)))
model.load_weights(weights_path)
# tf.status.expect_partial()
del model
sys.exit()

img_recon, gate_recon, means, stddev, z = model(images_np, mode=0)
img_recon = img_recon.numpy()
gate_recon = gate_recon.numpy()
z = z.numpy()

# de-normalization of gates and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon = cmvae_utils.dataset_utils.de_normalize_gate(gate_recon)

# if not read_table:
#     sys.exit()

# get stats for gate reconstruction
if read_table is True:
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
# plt.show()

# show interpolation btw two images in latent space
z_close = z[idx_close, :]
z_far = z[idx_far, :]
z_interp = cmvae_utils.geom_utils.interp_vector(z_close, z_far, num_interp_z)

# get the image predictions
img_recon_interp, gate_recon_interp = model.decode(z_interp, mode=0)
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

for idx in range(3):
    # axs[idx].grid()
    y_ticks_array = gate_recon_interp[:, idx][np.array([0, gate_recon_interp[:, idx].shape[0] - 1])]
    y_ticks_array = np.around(y_ticks_array, decimals=1)
    if idx > 0:
        y_ticks_array = y_ticks_array
    axs[idx].set_yticks(y_ticks_array)
    axs[idx].set_xticks(np.array([0, 9]))
    axs[idx].set_xticklabels((r'$I_a$', r'$I_b$'))

axs[0].set_title(r'$r$')
axs[1].set_title(r'$\theta$')
axs[2].set_title(r'$\psi$')

axs[0].set_ylabel('[meter]')
axs[1].set_ylabel(r'[deg]')
axs[2].set_ylabel(r'[deg]')

fig.savefig(os.path.join(output_dir, 'state_stats_interpolation_results.png'))

# plot the interpolated images
fig2 = plt.figure(figsize=(96, 96))
columns = num_interp_z + 2
rows = 1
fig2.add_subplot(rows, columns, 1)
img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[idx_close, :])
plt.axis('off')
plt.imshow(img_display)
for i in range(1, num_interp_z + 1):
    fig2.add_subplot(rows, columns, i + 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
    plt.axis('off')
    plt.imshow(img_display)
fig2.add_subplot(rows, columns, num_interp_z + 2)
img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[idx_far, :])
plt.axis('off')
plt.imshow(img_display)
fig2.savefig(os.path.join(output_dir, 'reconstruction_interpolation_results.png'))
# plt.show()

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
# plt.show()

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
