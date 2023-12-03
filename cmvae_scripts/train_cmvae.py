import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

import cmvae_models.cmvae
import cmvae_utils

cmvae_utils.dataset_utils.seed_environment()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Directory where the images/state data is contained', default="", type=str)
parser.add_argument('--model_dir', help='Directory where the pretrained model is', default="", type=str)
parser.add_argument('--n_z', help='Number of features to encode to', default=10, type=int)
parser.add_argument('--epochs', help='Number of epochs for the training run', default=30, type=int)
parser.add_argument('--use_cpu', help='Even with a cuda gpu enabled, force cpu instead', action='store_true')
parser.add_argument('--load_during_training', help='For large datasets, create the tensor slices per batch', action='store_true')
parser.add_argument('--max_size', help='Maximum number of images and state data to sample', default=None, type=int)
args = parser.parse_args()

if args.data_dir == '':
    print('No data directory specified, quitting!')
    quit()

device = []
if args.use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    device = tf.config.list_physical_devices('CPU')
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            device = tf.config.list_physical_devices('GPU')
        except RuntimeError as e:
            print(e)

print('Running on: {}'.format(device[0]))
data_dir = args.data_dir
output_dir = os.path.join(data_dir, 'results')

# check if output folder exists
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

pretrained_model_path = args.model_dir

# DEFINE TRAINING META PARAMETERS
batch_size = 32
epochs = args.epochs
n_z = args.n_z
latent_space_constraints = True
img_res = 64
learning_rate = 1e-4
load_during_training = args.load_during_training
mode = 0
max_size = args.max_size

# tf.config.run_functions_eagerly(True)

# CUSTOM TF FUNCTIONS
@tf.function
def calc_weighted_loss_img(img_recon, images_np):
    flat_pred = tf.reshape(img_recon, [-1])
    flat_gt = tf.reshape(images_np, [-1])
    error_sq = tf.math.squared_difference(flat_gt, flat_pred)
    softmax_weights = tf.math.exp(error_sq) / tf.reduce_sum(tf.math.exp(error_sq))
    weighted_error_sq = error_sq * softmax_weights
    loss = tf.reduce_sum(weighted_error_sq)
    return loss


def reset_metrics():
    train_loss_rec_img.reset_states()
    train_loss_rec_gate.reset_states()
    train_loss_kl.reset_states()
    test_loss_rec_img.reset_states()
    test_loss_rec_gate.reset_states()
    test_loss_kl.reset_states()


@tf.function
def regulate_weights(epoch):
    # for beta
    if epoch < 10.0:
        beta = 8.0
    else:
        beta = 8.0
    # t = 10
    # beta_min = 0.0  #0.000001
    # beta_max = 1.0  #0.0001
    # if epoch < t:
    #     # beta = beta_min + epoch/t*(beta_max-beta_min)
    #     beta = beta_max * 0.95**(t-epoch)  # ranges from 0.00592052922 to 0.95
    # else:
    #     beta = beta_max
    # for w_img
    if epoch < 100:
        w_img = 1.0
    else:
        w_img = 1.0
    # for w_gate
    if epoch < 100:
        w_gate = 1.0
    else:
        w_gate = 1.0
    return beta, w_img, w_gate


@tf.function
def compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode):
    # compute reconstruction loss
    if mode == 0:
        img_loss = tf.losses.mean_squared_error(img_gt, img_recon)
        # img_loss = tf.losses.mean_absolute_error(img_gt, img_recon)
        gate_loss = tf.losses.mean_squared_error(gate_gt, gate_recon)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + stddev - tf.math.pow(means, 2) - tf.math.exp(stddev)), axis=1))
    # elif mode == 1:
    #     # labels = tf.reshape(labels, predictions.shape)
    #     # recon_loss = tf.losses.mean_squared_error(labels, predictions)
    #     # recon_loss = loss_object(labels, predictions)
    # print('Predictions: {}'.format(predictions))
    # print('Labels: {}'.format(labels))
    # print('Lrec: {}'.format(recon_loss))
    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    return img_loss, gate_loss, kl_loss


@tf.function
def train(img_gt, gate_gt, epoch, mode):
    # freeze the non-utilized weights
    # if mode == 0:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = True
    #     model.p_gate.trainable = True
    # elif mode == 1:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = True
    #     model.p_gate.trainable = False
    # elif mode == 2:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = False
    #     model.p_gate.trainable = True
    with tf.GradientTape() as tape:
        img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
        img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
        img_loss = tf.reduce_mean(img_loss)
        gate_loss = tf.reduce_mean(gate_loss)
        beta, w_img, w_gate = regulate_weights(epoch)
        # weighted_loss_img = calc_weighted_loss_img(img_recon, img_gt)
        if mode == 0:
            total_loss = w_img * img_loss + w_gate * gate_loss + beta * kl_loss
            # total_loss = w_img * img_loss + beta * kl_loss
            # total_loss = weighted_loss_img + gate_loss + beta * kl_loss
            # total_loss = img_loss
            train_loss_rec_img.update_state(img_loss)
            train_loss_rec_gate.update_state(gate_loss)
            train_loss_kl.update_state(kl_loss)
        # TODO: later create structure for other training modes -- for now just training everything together
        # elif mode==1:
        #     total_loss = img_loss + beta*kl_loss
        #     train_kl_loss_m1(kl_loss)
        # elif mode==2:
        #     total_loss = gate_loss + beta*kl_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def test(img_gt, gate_gt, mode):
    img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        test_loss_rec_img.update_state(img_loss)
        test_loss_rec_gate.update_state(gate_loss)
        test_loss_kl.update_state(kl_loss)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if latent_space_constraints is True:
    model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=3, res=img_res, trainable_model=True)
else:
    model = cmvae_models.cmvae.Cmvae(n_z=n_z, gate_dim=3, res=img_res, trainable_model=True)

# create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# define metrics
train_loss_rec_img = tf.keras.metrics.Mean(name='train_loss_rec_img')
train_loss_rec_gate = tf.keras.metrics.Mean(name='train_loss_rec_gate')
train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
test_loss_rec_img = tf.keras.metrics.Mean(name='test_loss_rec_img')
test_loss_rec_gate = tf.keras.metrics.Mean(name='test_loss_rec_gate')
test_loss_kl = tf.keras.metrics.Mean(name='test_loss_kl')
metrics_writer = tf.summary.create_file_writer(output_dir)

img_train, img_test, dist_train, dist_test = cmvae_utils.dataset_utils.create_dataset_csv(data_dir, batch_size, img_res, max_size)

ds_train = None
ds_test = None

if not load_during_training:
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)

n_batches_train = (len(img_train) + batch_size - 1) // batch_size
n_batches_test = (len(img_test) + batch_size - 1) // batch_size

lowest_loss = np.Inf
# train
print('Start training ...')
for epoch in range(epochs):
    if not load_during_training:
        for train_images, train_labels in ds_train:
            train(train_images, train_labels, epoch, mode)
        for test_images, test_labels in ds_test:
            test(test_images, test_labels, mode)
    else:
        img_train_cpy = img_train
        dist_train_cpy = dist_train

        img_test_cpy = img_test
        dist_test_cpy = dist_test

        for _ in range(n_batches_train):
            ds_train = tf.data.Dataset.from_tensor_slices((img_train_cpy[:batch_size], dist_train_cpy[:batch_size])).batch(batch_size)
            for train_images, train_labels in ds_train:
                train(train_images, train_labels, epoch, mode)
                if len(img_train_cpy) > batch_size:
                    img_train_cpy = img_train_cpy[batch_size:]
                    dist_train_cpy = dist_train_cpy[batch_size:]

        for _ in range(n_batches_test):
            ds_test = tf.data.Dataset.from_tensor_slices((img_test_cpy[:batch_size], dist_test_cpy[:batch_size])).batch(batch_size)
            for test_images, test_labels in ds_test:
                test(test_images, test_labels, mode)
                if len(img_test_cpy) > batch_size:
                    img_test_cpy = img_test_cpy[batch_size:]
                    dist_test_cpy = dist_test_cpy[batch_size:]

    # save model
    if epoch % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(output_dir))
        model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(epoch)))

    if mode == 0:
        with metrics_writer.as_default():
            train_total_loss = train_loss_rec_img.result() + train_loss_rec_gate.result() + train_loss_kl.result()
            test_total_loss = test_loss_rec_img.result() + test_loss_rec_gate.result() + test_loss_kl.result()

            tf.summary.scalar('train/loss_rec_img', train_loss_rec_img.result(), step=epoch)
            tf.summary.scalar('train/loss_rec_gate', train_loss_rec_gate.result(), step=epoch)
            tf.summary.scalar('train/loss_kl', train_loss_kl.result(), step=epoch)
            tf.summary.scalar('train/total_loss', train_total_loss, step=epoch)
            tf.summary.scalar('test/loss_rec_img', test_loss_rec_img.result(), step=epoch)
            tf.summary.scalar('test/loss_rec_gate', test_loss_rec_gate.result(), step=epoch)
            tf.summary.scalar('test/loss_kl', test_loss_kl.result(), step=epoch)
            tf.summary.scalar('test/total_loss', test_total_loss, step=epoch)

            if epoch == 0:
                with open(os.path.join(curr_dir, os.pardir, 'launchers', 'train_output.txt'), 'w') as log_file:
                    log_file.write('Tensorboard event file: {}\n'.format(output_dir))

        print('Epoch {} | TRAIN: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {}'
              .format(epoch, train_loss_rec_img.result(), train_loss_rec_gate.result(), train_loss_kl.result(), train_total_loss,
                      test_loss_rec_img.result(), test_loss_rec_gate.result(), test_loss_kl.result(), test_total_loss))
        if test_total_loss < lowest_loss:
            print('Best model found, total test loss: {}. Saving weights to {}'.format(test_total_loss, output_dir))
            model.save_weights(os.path.join(output_dir, 'best_model.ckpt'))
            lowest_loss = test_total_loss

        reset_metrics()  # reset all the accumulators of metrics

print('End of training')
model.save_weights(os.path.join(output_dir, 'cmvae_model_{}.ckpt'.format(epochs - 1)))
