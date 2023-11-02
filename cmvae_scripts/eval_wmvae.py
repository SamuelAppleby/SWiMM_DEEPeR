"""
WORLD MODELS VAE VERSION OF EVAL_CMVAE.PY
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import cmvae_models.wmvae
import cmvae_utils

# define testing meta parameters
data_dir = '../logs/vae_test_data'
output_dir = '../logs/cmvae/300k_run_17_08_22_wm/'
model_to_eval = 'cmvae_model_25.ckpt'

n_z = 10
img_res = 64
learning_rate = 1e-4
beta = 8.0
num_imgs_display = 50
columns = 10
rows = 10
read_table = True

num_interp_z = 10
idx_close = 340  # NB: image want to use is 341.png but index for images_np is 340 with being zero-indexed
idx_far = 761  # same as above

z_range_mural = [-0.02, 0.02]
z_num_mural = 11

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
model = cmvae_models.wmvae.WmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=False)

# load trained weights from checkpoint
print('Loading weights from {}'.format(os.path.join(output_dir, model_to_eval)))
model.load_weights(os.path.join(output_dir, model_to_eval))

# initialize dataset and dataset iterator
model.sess.run(model.init_op, feed_dict={model.img_data: images_np, model.state_data: raw_table, model.batch_size: 1})

pbar = tqdm(total=len(raw_table))
# initialise output variables with first run of graph
z, img_recon = model.sess.run([model.z, model.img_recon])
pbar.update(1)
# concat output to initial variables with each subsequent run of graph
for _ in range(len(raw_table) - 1):
    latent_vector, image_reconstruction = model.sess.run([
        model.z,
        model.img_recon
    ])
    z = np.concatenate([z, latent_vector], axis=0)
    img_recon = np.concatenate([img_recon, image_reconstruction], axis=0)
    pbar.update(1)
pbar.close()

# de-normalization of states and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)

# show some reconstruction figures
fig = plt.figure(figsize=(20, 20))
# create array of num_imgs_display indexes to use with random sampler rather than using index 0 to num_imgs_display so more variation
imgs_to_use = np.random.choice(range(len(raw_table)), num_imgs_display, replace=False)
for i in range(1, num_imgs_display + 1):
    img_to_use = imgs_to_use[i - 1]
    idx_orig = (i - 1) * 2 + 1
    fig.add_subplot(rows, columns, idx_orig)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(images_np[img_to_use, :])
    plt.axis('off')
    plt.imshow(img_display)
    fig.add_subplot(rows, columns, idx_orig + 1)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon[img_to_use, :])
    plt.axis('off')
    plt.imshow(img_display)
fig.savefig(os.path.join(output_dir, 'reconstruction_results.png'))
plt.show()

# show interpolation btw two images in latent space
z_close = z[idx_close, :]
z_far = z[idx_far, :]
z_interp = cmvae_utils.geom_utils.interp_vector(z_close, z_far, num_interp_z)

# get the image predictions
# add dimension to front so (1,10) not (10,1)
z_feed = z_interp[0].reshape(1, -1)
# initialise output variable with first call to decode
img_recon_interp = model.decode(z_feed)
# concat output to initial variable with each subsequent call to decode
for i in range(1, len(z_interp)):
    z_feed = z_interp[i].reshape(1, -1)
    x = model.decode(z_feed)
    img_recon_interp = np.concatenate([img_recon_interp, x], axis=0)

# de-normalization of states and images
img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)

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
plt.show()

# new plot traveling through latent space
fig3 = plt.figure(figsize=(96, 96))
columns = z_num_mural
rows = n_z
z_values = cmvae_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
for i in range(1, z_num_mural * n_z + 1):
    fig3.add_subplot(rows, columns, i)
    z = np.zeros((1, n_z)).astype(np.float32)
    z[0, int((i - 1) / columns)] = z_values[i % columns - 1]
    img_recon_interp = model.decode(z)
    img_recon_interp = ((img_recon_interp[0, :] + 1.0) / 2.0 * 255.0).astype(np.uint8)
    img_display = cmvae_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
    plt.axis('off')
    plt.imshow(img_display)
fig3.savefig(os.path.join(output_dir, 'z_mural.png'))
plt.show()
