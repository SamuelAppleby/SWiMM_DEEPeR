import os
import yaml

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_global_config.yml'), 'r') as f:
    cmvae_global_config = yaml.load(f, Loader=yaml.UnsafeLoader)

import tensorflow as tf

if cmvae_global_config['use_cpu_only']:
    tf.config.set_visible_devices([], 'GPU')

from tqdm import tqdm

import cmvae_utils.dataset_utils
from gym_underwater.utils import load_environment_config, load_cmvae_training_config, output_devices, count_directories_in_directory, parse_command_args, \
    tensorflow_seeding, duplicate_directory, load_cmvae, output_command_line_arguments

cmvae = load_cmvae(cmvae_global_config=cmvae_global_config)

env_config = load_environment_config(project_dir)

parse_command_args(env_config)

tensorflow_seeding(env_config['seed'])

cmvae_training_config = load_cmvae_training_config(project_dir)

assert (cmvae_training_config['train_dir'] != '') and os.path.isdir(cmvae_training_config['train_dir']), 'Require valid image/state_data directory'

output_dir = os.path.join(project_dir, 'models', 'cmvae')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = os.path.join(output_dir, str(count_directories_in_directory(output_dir)))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# DEFINE TRAINING META PARAMETERS
batch_size = cmvae_training_config['batch_size']
epochs = cmvae_training_config['epochs']
n_z = cmvae_global_config['n_z']
latent_space_constraints = cmvae_global_config['latent_space_constraints']
img_res = tuple(cmvae_global_config['img_res'])
learning_rate = float(cmvae_training_config['learning_rate'])
load_during_training = cmvae_training_config['load_during_training']
mode = 0
max_size = cmvae_training_config['max_size']
window_size = cmvae_training_config['window_size']
loss_threshold = cmvae_training_config['loss_threshold']


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
        return img_loss, gate_loss, kl_loss
    # elif mode == 1:
    #     # labels = tf.reshape(labels, predictions.shape)
    #     # recon_loss = tf.losses.mean_squared_error(labels, predictions)
    #     # recon_loss = loss_object(labels, predictions)
    # print('Predictions: {}'.format(predictions))
    # print('Labels: {}'.format(labels))
    # print('Lrec: {}'.format(recon_loss))
    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))


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
        img_recon, gate_recon, means, stddev, z = cmvae(img_gt, training=True, mode=mode)
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
        gradients = tape.gradient(total_loss, cmvae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cmvae.trainable_variables))


@tf.function
def test(img_gt, gate_gt, mode):
    img_recon, gate_recon, means, stddev, z = cmvae(img_gt, training=False, mode=mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        test_loss_rec_img.update_state(img_loss)
        test_loss_rec_gate.update_state(gate_loss)
        test_loss_kl.update_state(kl_loss)


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

with metrics_writer.as_default():
    tf.summary.scalar('time', 0, step=0)

img_train, img_test, dist_train, dist_test = cmvae_utils.dataset_utils.create_dataset_csv(cmvae_training_config['train_dir'], img_res, max_size, env_config['seed'])

ds_train = None
ds_test = None

if not load_during_training:
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)

n_batches_train = (len(img_train) + batch_size - 1) // batch_size
n_batches_test = (len(img_test) + batch_size - 1) // batch_size

lowest_loss = 1e6
current_window_loss = 1e6
bad_epochs = 0

# train
print('Start training ...')
for epoch in tqdm(range(epochs)):
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

        for _ in tqdm(range(n_batches_train)):
            ds_train = tf.data.Dataset.from_tensor_slices((img_train_cpy[:batch_size], dist_train_cpy[:batch_size])).batch(batch_size)
            for train_images, train_labels in ds_train:
                train(train_images, train_labels, epoch, mode)
                if len(img_train_cpy) > batch_size:
                    img_train_cpy = img_train_cpy[batch_size:]
                    dist_train_cpy = dist_train_cpy[batch_size:]

        for _ in tqdm(range(n_batches_test)):
            ds_test = tf.data.Dataset.from_tensor_slices((img_test_cpy[:batch_size], dist_test_cpy[:batch_size])).batch(batch_size)
            for test_images, test_labels in ds_test:
                test(test_images, test_labels, mode)
                if len(img_test_cpy) > batch_size:
                    img_test_cpy = img_test_cpy[batch_size:]
                    dist_test_cpy = dist_test_cpy[batch_size:]
    # save model
    if (((epoch + 1) % 5) == 0) or (epoch + 1 == epochs):
        cmvae.save_weights(os.path.join(output_dir, 'epoch_{}'.format(epoch), 'model.ckpt'))

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

            print('Epoch {} | TRAIN: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {}'
                  .format(epoch, train_loss_rec_img.result(), train_loss_rec_gate.result(), train_loss_kl.result(), train_total_loss,
                          test_loss_rec_img.result(), test_loss_rec_gate.result(), test_loss_kl.result(), test_total_loss))

            if test_total_loss < lowest_loss:
                print('Best model found, total test loss: {}. Saving weights to {}'.format(test_total_loss, output_dir))
                cmvae.save_weights(os.path.join(output_dir, 'best_model', 'model.ckpt'))
                if window_size is not None:
                    if ((current_window_loss - test_total_loss) / current_window_loss) >= loss_threshold:
                        bad_epochs = 0
                        current_window_loss = test_total_loss
                        print('Checkpoint reset, require next loss of: {}%'.format((current_window_loss * (1 - loss_threshold))))
                    else:
                        bad_epochs += 1
                lowest_loss = test_total_loss
            elif window_size is not None:
                bad_epochs += 1

            if (window_size is not None) and (bad_epochs == window_size):
                print(f'No better loss after: {bad_epochs} epochs')
                cmvae.save_weights(os.path.join(output_dir, 'epoch_{}'.format(epoch), 'model.ckpt'))
                break

        reset_metrics()  # reset all the accumulators of metrics

with metrics_writer.as_default():
    tf.summary.scalar('time', 0, step=1)

config_dir = os.path.join(output_dir, 'configs')
duplicate_directory(os.path.join(project_dir, 'configs'), config_dir, dirs_to_exclude=['hyperparams'], files_to_exclude=['cmvae_inference_config.yml', 'callbacks.yml', 'env_wrapper.yml', 'server_config.json'])
output_devices(config_dir, tensorflow_device=True)
output_command_line_arguments(config_dir)
