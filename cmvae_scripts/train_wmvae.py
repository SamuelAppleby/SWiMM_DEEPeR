'''
WORLD MODELS VAE VERSION OF TRAIN_CMVAE.PY
'''

import tensorflow as tf
import os
import sys
from tqdm import tqdm
curr_dir = os.path.dirname(os.path.abspath(__file__))

# imports
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import cmvae_models.wmvae
import cmvae_utils

# define training meta parameters
data_dir = '../logs/vae_data_300k'
output_dir = '../logs/cmvae/300k_run_17_08_22_wm'
big_data = True
batch_size = 32
epochs = 30 #15 #50
n_z = 10
img_res = 64
max_size = None  
learning_rate = 1e-4
beta = 8.0

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
    model = cmvae_models.wmvae.WmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=True)
else:
    model = cmvae_models.wmvae.WmvaeDirect(n_z=n_z, state_dim=3, res=img_res, learning_rate=learning_rate, beta=beta, trainable_model=True, big_data=False) 

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
        (train_img_loss, train_kl_loss, train_total_loss, global_step, _) = model.sess.run([
            model.img_loss,
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
        (test_img_loss, test_kl_loss, test_total_loss) = model.sess.run([
            model.img_loss,
            model.kl_loss,
            model.total_loss
        ])
        pbar.update(1)
    pbar.close()

    # save model
    if epoch % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(output_dir))
        model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(epoch))) 

    # write to tensorboard
    train_img_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss images", simple_value=train_img_loss)])
    metrics_writer.add_summary(train_img_summary, epoch)
    #train_state_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss state", simple_value=train_state_loss)])
    #metrics_writer.add_summary(train_state_summary, epoch)
    train_summary = tf.Summary(value=[tf.Summary.Value(tag="Training loss", simple_value=train_total_loss)])
    metrics_writer.add_summary(train_summary, epoch)
    test_img_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss images", simple_value=test_img_loss)])
    metrics_writer.add_summary(test_img_summary, epoch)
    #test_state_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss state", simple_value=test_state_loss)])
    #metrics_writer.add_summary(test_state_summary, epoch)
    test_summary = tf.Summary(value=[tf.Summary.Value(tag="Validation loss", simple_value=test_total_loss)])
    metrics_writer.add_summary(test_summary, epoch)
  
    print('Epoch {} | TRAIN: L_img: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_kl: {}, L_tot: {}'
            .format(epoch, train_img_loss, train_kl_loss, train_total_loss, test_img_loss, test_kl_loss, test_total_loss))

print('End of training, saving final model')
model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(epoch)))
