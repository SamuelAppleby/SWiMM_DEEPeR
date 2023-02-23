'''
file: eval_cmvae.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import random
import yaml

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

import cmvae_models.cmvae
import cmvae_utils

# define testing meta parameters
data_dir = '/home/campus.ncl.ac.uk/b3024896/Downloads/64x64_50'
output_dir = '/home/campus.ncl.ac.uk/b3024896/Downloads/cmvae_02_16_2023_nz_10/'
model_to_eval = 'cmvae_model_49.ckpt'

n_z = 10
img_res = 64
learning_rate = 1e-4
beta = 8.0
num_imgs_display = 50
columns = 10
rows = 10
read_table = True

# num_interp_z = 10
# idx_close = 2385 #NB: image filename minus 1
# idx_far = 2768

# z_range_mural = [-0.02, 0.02]
# z_num_mural = 11

with open('../Configs/env/config.yml', 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)

# seeding for reproducability
seed = env_config['seed']
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load test raw data
images_np, raw_table = cmvae_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res, read_table=read_table)
print('Done with dataset')

###### DEBUGGING ########

# test_image = ((images_np[1] + 1.0) / 2.0 * 255.0).astype(np.uint8)

# cv2.imshow('test', test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()  

# print(raw_table[1])

#########################

# create model
model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=False) 

# load trained weights from checkpoint
print('Loading weights from {}'.format(os.path.join(output_dir, model_to_eval)))
model.load_weights(os.path.join(output_dir, model_to_eval))

# initialize dataset and dataset iterator
model.sess.run(model.init_op, feed_dict={model.img_data: images_np, model.state_data: raw_table, model.batch_size:1})

pbar = tqdm(total=len(raw_table))
# initialise output variables with first run of graph
z, img_recon, state_recon = model.sess.run([model.z, model.img_recon, model.state_recon])
pbar.update(1)
# concat output to initial variables with each subsequent run of graph
for _ in range(len(raw_table)-1):
    latent_vector, image_reconstruction, state_prediction = model.sess.run([
        model.z,
        model.img_recon,
        model.state_recon
    ])
    z = np.concatenate([z, latent_vector], axis=0)
    img_recon = np.concatenate([img_recon, image_reconstruction], axis=0)
    state_recon = np.concatenate([state_recon, state_prediction], axis=0)
    pbar.update(1)
pbar.close()

# de-normalization of states and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)
state_recon = cmvae_utils.dataset_utils.de_normalize_state(state_recon)

# # get stats for state predictions
# cmvae_utils.stats_utils.calculate_state_stats(state_recon, raw_table, output_dir)

# show some reconstruction figures
fig = plt.figure(figsize=(20, 20))
# create array of num_imgs_display indexes to use with random sampler rather than using index 0 to num_imgs_display so more variation
imgs_to_use = np.random.choice(range(len(raw_table)), num_imgs_display, replace=False)
for i in range(1, num_imgs_display+1):
    img_to_use = imgs_to_use[i-1]
    idx_orig = (i-1)*2+1
    fig.add_subplot(rows, columns, idx_orig)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[img_to_use, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig.add_subplot(rows, columns, idx_orig+1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon[img_to_use, :])
    plt.axis('off')
    plt.imshow(img_display)
fig.savefig(os.path.join(output_dir, 'reconstruction_results.png'))
plt.show()

# # show interpolation btw two images in latent space
# z_close = z[idx_close, :]
# z_far = z[idx_far, :]
# z_interp = cmvae_utils.geom_utils.interp_vector(z_close, z_far, num_interp_z)

# # get the image predictions
# # add dimension to front so (1,10) not (10,)
# z_feed = z_interp[0].reshape(1, -1)
# # initialise output variable with first call to decode
# img_recon_interp, state_recon_interp = model.decode(z_feed)
# # concat output to initial variable with each subsequent call to decode
# for i in range(1, len(z_interp)):
#     z_feed = z_interp[i].reshape(1, -1)
#     x, y = model.decode(z_feed)
#     img_recon_interp = np.concatenate([img_recon_interp, x], axis=0)
#     state_recon_interp = np.concatenate([state_recon_interp, y], axis=0)

# # de-normalization of states and images
# img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
# state_recon_interp = cmvae_utils.dataset_utils.de_normalize_state(state_recon_interp)

# # join predictions with array and print
# indices = np.array([np.arange(num_interp_z)]).transpose()
# results = np.concatenate((indices, state_recon_interp), axis=1)
# print('Img index | Predictions: = \n{}'.format(results))

# fig, axs = plt.subplots(1, 3, tight_layout=True)
# axs[0].plot(np.arange(state_recon_interp.shape[0]), state_recon_interp[:, 0], 'b-', label='r')
# axs[1].plot(np.arange(state_recon_interp.shape[0]), state_recon_interp[:, 1], 'b-', label=r'$\theta$') 
# axs[2].plot(np.arange(state_recon_interp.shape[0]), state_recon_interp[:, 2], 'b-', label=r'$\psi$')

# for idx in range(3): 
#     y_ticks_array = state_recon_interp[:, idx][np.array([0, state_recon_interp[:, idx].shape[0]-1])]
#     y_ticks_array = np.around(y_ticks_array, decimals=1)
#     if idx > 0:
#         y_ticks_array = y_ticks_array 
#     axs[idx].set_yticks(y_ticks_array)
#     axs[idx].set_xticks(np.array([0, 9]))
#     axs[idx].set_xticklabels((r'$I_a$', r'$I_b$'))

# axs[0].set_title(r'$r$')
# axs[1].set_title(r'$\theta$')
# axs[2].set_title(r'$\psi$')

# axs[0].set_ylabel('[meter]')
# axs[1].set_ylabel(r'[deg]')
# axs[2].set_ylabel(r'[deg]') 

# fig.savefig(os.path.join(output_dir, 'state_stats_interpolation_results.png'))

# # plot the interpolated images
# fig2 = plt.figure(figsize=(96, 96))
# columns = num_interp_z + 2
# rows = 1
# fig2.add_subplot(rows, columns, 1)
# img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[idx_close, :])
# plt.axis('off')
# plt.imshow(img_display)
# for i in range(1, num_interp_z + 1):
#     fig2.add_subplot(rows, columns, i+1)
#     img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
#     plt.axis('off')
#     plt.imshow(img_display)
# fig2.add_subplot(rows, columns, num_interp_z + 2)
# img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[idx_far, :])
# plt.axis('off')
# plt.imshow(img_display)
# fig2.savefig(os.path.join(output_dir, 'reconstruction_interpolation_results.png'))
# plt.show()

# # new plot traveling through latent space
# fig3 = plt.figure(figsize=(96, 96))
# columns = z_num_mural
# rows = n_z
# z_values = cmvae_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
# for i in range(1, z_num_mural*n_z + 1):
#     fig3.add_subplot(rows, columns, i)
#     z = np.zeros((1, n_z)).astype(np.float32)
#     z[0, int((i-1)/columns)] = z_values[i%columns-1] 
#     img_recon_interp, state_recon_interp = model.decode(z)
#     img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
#     img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
#     plt.axis('off')
#     plt.imshow(img_display)
# fig3.savefig(os.path.join(output_dir, 'z_mural.png'))
# plt.show()

# # single-channel version of above 
# fig4 = plt.figure(figsize=(96, 96))
# columns = z_num_mural
# rows = 1
# z_values = cmvae_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
# for i in range(1, z_num_mural + 1):
#     fig4.add_subplot(rows, columns, i)
#     z = np.zeros((1, n_z)).astype(np.float32)
#     z[0, 2] = z_values[i-1] # interp across yaw feature
#     z[0, 0] = 0.02 # but hard code distance feature to 'close'
#     img_recon_interp, state_recon_interp = model.decode(z)
#     img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
#     img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
#     plt.axis('off')
#     plt.imshow(img_display)
# fig4.savefig(os.path.join(output_dir, 'yaw_up_close.png'))
# plt.show()




