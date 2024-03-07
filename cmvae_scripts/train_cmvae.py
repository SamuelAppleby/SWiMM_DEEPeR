import os
import shutil
import yaml
from tqdm import tqdm

import cmvae_models.cmvae
import cmvae_utils.dataset_utils

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'cmvae_config.yml'), 'r') as f:
    cmvae_config = yaml.load(f, Loader=yaml.UnsafeLoader)

if cmvae_config['train_dir'] == '':
    print('No data directory specified, quitting!')
    quit()

if cmvae_config['use_cpu']:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
else:
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'config.yml'), 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    tf.keras.utils.set_random_seed(env_config['seed'])

print('Devices: {}'.format(tf.config.list_physical_devices()))

train_dir = cmvae_config['train_dir']
output_dir = os.path.join(train_dir, 'results_seed_{}_device_{}'.format(env_config['seed'], 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu'))

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

# DEFINE TRAINING META PARAMETERS
batch_size = cmvae_config['batch_size']
epochs = cmvae_config['epochs']
n_z = cmvae_config['n_z']
latent_space_constraints = cmvae_config['latent_space_constraints']
img_res = cmvae_config['img_res']
learning_rate = float(cmvae_config['learning_rate'])
load_during_training = cmvae_config['load_during_training']
mode = 0
max_size = cmvae_config['max_size']
window_size = cmvae_config['window_size']
loss_threshold = cmvae_config['loss_threshold']


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
        img_recon, gate_recon, means, stddev, z = model(img_gt, training=True, mode=mode)
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
    img_recon, gate_recon, means, stddev, z = model(img_gt, training=False, mode=mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        test_loss_rec_img.update_state(img_loss)
        test_loss_rec_gate.update_state(gate_loss)
        test_loss_kl.update_state(kl_loss)


if latent_space_constraints:
    model = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, seed=env_config['seed'])
else:
    model = cmvae_models.cmvae.Cmvae(n_z=n_z, gate_dim=3)

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

img_train, img_test, dist_train, dist_test = cmvae_utils.dataset_utils.create_dataset_csv(train_dir, model.img_res, max_size, env_config['seed'])

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
        model.save_weights(os.path.join(output_dir, 'cmvae_model_{}.ckpt'.format(epoch)))

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
                with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'launchers', 'train_output.txt'), 'w') as log_file:
                    log_file.write('Tensorboard event file: {}\n'.format(output_dir))

            print('Epoch {} | TRAIN: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {}'
                  .format(epoch, train_loss_rec_img.result(), train_loss_rec_gate.result(), train_loss_kl.result(), train_total_loss,
                          test_loss_rec_img.result(), test_loss_rec_gate.result(), test_loss_kl.result(), test_total_loss))

            if test_total_loss < lowest_loss:
                print('Best model found, total test loss: {}. Saving weights to {}'.format(test_total_loss, output_dir))
                model.save_weights(os.path.join(output_dir, 'best_model.ckpt'))
                if window_size is not None:
                    if ((current_window_loss - test_total_loss) / current_window_loss) > loss_threshold:
                        bad_epochs = 0
                        current_window_loss = test_total_loss
                        print('Checkpoint reset, require next loss of: {}%'.format((current_window_loss * (1 - loss_threshold))))
                    else:
                        bad_epochs += 1
                lowest_loss = test_total_loss
            elif window_size is not None:
                bad_epochs += 1

            if (window_size is not None) and (bad_epochs == window_size):
                print('No better loss after: {} epochs'.format(bad_epochs))
                model.save_weights(os.path.join(output_dir, 'cmvae_model_{}.ckpt'.format(epoch)))
                break

        reset_metrics()  # reset all the accumulators of metrics

print('End of training')
