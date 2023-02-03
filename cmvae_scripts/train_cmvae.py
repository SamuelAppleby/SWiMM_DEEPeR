'''
file: train_cmvae.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import tensorflow as tf
import os
import sys
from tqdm import tqdm
import numpy as np
import random 
import yaml

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

import cmvae_models.cmvae
import cmvae_utils

# define training meta parameters
data_dir = 'D:' + os.sep + 'vae' + os.sep + '1920x1080'
output_dir = 'D:' + os.sep + 'vae' + os.sep + '1920x1080' + os.sep + 'cmvae_run_10_01_23'
pretrained_model_path = 'D:' + os.sep + 'vae' + os.sep + '1920x1080' + os.sep + 'cmvae_run_10_01_23' + 'cmvae_model_29.ckpt'
#data_dir = '/home/campus.ncl.ac.uk/b3024896/Downloads/dummy_images'
#output_dir = '/home/campus.ncl.ac.uk/b3024896/Projects/RLNet/Logs/vae/1920x1080/cmvae_run_10_01_23'
#pretrained_model_path = '/home/campus.ncl.ac.uk/b3024896/Projects/RLNet/Logs/vae/1920x1080/cmvae_run_10_01_23/cmvae_model_29.ckpt'
big_data = False
batch_size = 32
epochs = 30 #15 #50
n_z = 10
img_res = 64
max_size = None  
learning_rate = 1e-4
beta = 8.0

with open('../Configs/env/config.yml', 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)

# seeding for reproducability
seed = env_config['seed']
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load dataset
# for 10k and 50k datasets, the original create_dataset_csv is sufficient, for the 300k, cannot load all 300k images into a numpy array
# so wrote a new function that just returns filenames and cmvae.py then loads images in batches on the fly
print('Starting dataset')
if big_data:
    train_ds, test_ds, n_batches_train, n_batches_test = cmvae_utils.dataset_utils.create_dataset_filepaths(data_dir, batch_size, img_res, max_size=max_size)
else:
    train_ds, test_ds, n_batches_train, n_batches_test = cmvae_utils.dataset_utils.create_dataset_csv(data_dir, batch_size, img_res, max_size=max_size)
print('Done with dataset')

# create model
if big_data:
    model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=True)
else:
    model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=False) 

# check if training on top of existing model and if so load weights
if pretrained_model_path != '':
    print('Loading weights from {}'.format(pretrained_model_path))
    model.load_weights(pretrained_model_path)
    spliced_path = pretrained_model_path.rsplit('_')[-1]
    num_pretrained_epochs = int(spliced_path.rsplit('.')[0]) + 1

# check if output folder exists
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# create tensorboard writer
metrics_writer = tf.summary.FileWriter(output_dir, model.graph)

# train
for epoch in range(epochs):
    # initialize dataset iterator with train data
    model.sess.run(model.init_op, feed_dict={model.img_data: train_ds[0], model.state_data: train_ds[1], model.batch_size:batch_size})
    pbar = tqdm(total=n_batches_train)
    for _ in range(n_batches_train):
        (train_img_loss, train_state_loss, train_kl_loss, train_total_loss, global_step, _) = model.sess.run([
            model.img_loss,
            model.state_loss,
            model.kl_loss,
            model.total_loss,
            model.global_step,
            model.train_op
        ])
        pbar.update(1)
    pbar.close()
    # initialize dataset iterator with test data
    model.sess.run(model.init_op, feed_dict={model.img_data: test_ds[0], model.state_data: test_ds[1], model.batch_size:batch_size})
    pbar = tqdm(total=n_batches_test)
    for _ in range(n_batches_test):
        (test_img_loss, test_state_loss, test_kl_loss, test_total_loss) = model.sess.run([
            model.img_loss,
            model.state_loss,
            model.kl_loss,
            model.total_loss
        ])
        pbar.update(1)
    pbar.close()

    # calc total epochs if training on top of pretrained model
    if pretrained_model_path != '':
        total_epochs = num_pretrained_epochs + epoch
    else:
        total_epochs = epoch

    # save model
    if total_epochs % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(output_dir))
        model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(total_epochs))) 

    # write to tensorboard
    train_img_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss images", simple_value=train_img_loss)])
    metrics_writer.add_summary(train_img_summary, total_epochs)
    train_state_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss state", simple_value=train_state_loss)])
    metrics_writer.add_summary(train_state_summary, total_epochs)
    train_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss", simple_value=train_total_loss)])
    metrics_writer.add_summary(train_summary, total_epochs)
    test_img_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss images", simple_value=test_img_loss)])
    metrics_writer.add_summary(test_img_summary, total_epochs)
    test_state_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss state", simple_value=test_state_loss)])
    metrics_writer.add_summary(test_state_summary, total_epochs)
    test_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss", simple_value=test_total_loss)])
    metrics_writer.add_summary(test_summary, total_epochs)
  
    print('Epoch {} | TRAIN: L_img: {}, L_state: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_state: {}, L_kl: {}, L_tot: {}'
            .format(total_epochs, train_img_loss, train_state_loss, train_kl_loss, train_total_loss, 
            test_img_loss, test_state_loss, test_kl_loss, test_total_loss))

print('End of training, saving final model')
model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(total_epochs)))
