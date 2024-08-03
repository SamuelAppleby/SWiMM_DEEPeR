import tensorflow as tf
from keras.backend import random_normal
from keras.layers import Lambda

from . import decoders
from . import dronet
from . import transformer


# model definition class
class Cmvae(tf.keras.Model):
    def __init__(self, n_z, gate_dim=3):
        super(Cmvae, self).__init__()
        # create the 3 base models:
        self.q_img = dronet.Dronet(num_outputs=n_z * 2, include_top=True)
        self.p_img = decoders.ImgDecoder()
        self.p_gate = decoders.GateDecoder(gate_dim=gate_dim)
        # Create sampler
        self.mean_params = Lambda(lambda x: x[:, : n_z])
        self.stddev_params = Lambda(lambda x: x[:, n_z:])

    @tf.function
    def call(self, x, mode, training=False):
        # Possible modes for reconstruction:
        # 0: img -> img + gate
        # 1: img -> img
        # 2: img -> gate
        z, means, stddev = self.encode(x, training=training)
        img_recon, gate_recon = self.decode(z, mode, training=training)
        return img_recon, gate_recon, means, stddev, z

    @tf.function
    def encode(self, x, training=False):
        x = self.q_img(x, training=training)
        means = self.mean_params(x, training=training)
        stddev = tf.math.exp(0.5 * self.stddev_params(x, training=training))
        eps = random_normal(tf.shape(stddev))
        z = means + eps * stddev
        return z, means, stddev

    @tf.function
    def decode(self, z, mode, training=False):
        # Possible modes for reconstruction:
        # 0: z -> img + gate
        # 1: z -> img
        # 2: z -> gate
        if mode == 0:
            img_recon = self.p_img(z, training=training)
            gate_recon = self.p_gate(z, training=training)
            return img_recon, gate_recon
        elif mode == 1:
            img_recon = self.p_img(z, training=training)
            gate_recon = False
            return img_recon, gate_recon
        elif mode == 2:
            img_recon = False
            gate_recon = self.p_gate(z, training=training)
            return img_recon, gate_recon


# model definition class. This is the benchmarked and tested one for the pipeline.
class CmvaeDirect(tf.keras.Model):
    def __init__(self, n_z, img_res=None):
        super(CmvaeDirect, self).__init__()
        # create the base models:
        self.q_img = dronet.Dronet(num_outputs=n_z * 2, include_top=True)
        self.p_img = decoders.ImgDecoder()
        self.p_R = transformer.NonLinearTransformer()
        self.p_Theta = transformer.NonLinearTransformer()
        self.p_Psi = transformer.NonLinearTransformer()
        # Create sampler
        self.mean_params = tf.keras.layers.Lambda(lambda x: x[:, : n_z])
        self.stddev_params = tf.keras.layers.Lambda(lambda x: x[:, n_z:])
        self.R_params = tf.keras.layers.Lambda(lambda x: x[:, 0])
        self.Theta_params = tf.keras.layers.Lambda(lambda x: x[:, 1])
        self.Psi_params = tf.keras.layers.Lambda(lambda x: x[:, 2])
        self.img_res = img_res

    @tf.function
    def call(self, x, mode, training=False):
        # Possible modes for reconstruction:
        # 0: img -> img + gate
        # 1: img -> img
        # 2: img -> gate
        z, means, stddev = self.encode(x, training=training)
        img_recon, gate_recon = self.decode(z, mode, training=training)
        return img_recon, gate_recon, means, stddev, z

    @tf.function
    def encode(self, x, training=False):
        x = self.q_img(x, training=training)
        means = self.mean_params(x, training=training)
        stddev = tf.math.exp(0.5 * self.stddev_params(x, training=training))
        eps = tf.keras.backend.random_normal(tf.shape(stddev))
        z = means + eps * stddev
        return z, means, stddev

    @tf.function
    def decode(self, z, mode, training=False):
        # Possible modes for reconstruction:
        # 0: z -> img + gate
        # 1: z -> img
        # 2: z -> gate
        if mode == 0:
            img_recon = self.p_img(z, training=training)
            r_params, theta_params, psi_params = self.extract_gate_params(z, training=training)
            gate_recon = tf.keras.layers.concatenate([self.p_R(r_params, training=training), self.p_Theta(theta_params, training=training), self.p_Psi(psi_params, training=training)], axis=1)
            return img_recon, gate_recon
        elif mode == 1:
            img_recon = self.p_img(z, training=training)
            gate_recon = False
            return img_recon, gate_recon
        elif mode == 2:
            img_recon = False
            r_params, theta_params, psi_params = self.extract_gate_params(z, training=training)
            gate_recon = tf.keras.layers.concatenate([self.p_R(r_params, training=training), self.p_Theta(theta_params, training=training), self.p_Psi(psi_params, training=training)], axis=1)
            return img_recon, gate_recon

    @tf.function
    def extract_gate_params(self, z, training=False):
        # extract part of z vector
        r_params = self.R_params(z, training=training)
        theta_params = self.Theta_params(z, training=training)
        psi_params = self.Psi_params(z, training=training)
        # reshape variables
        r_params = tf.reshape(r_params, [r_params.shape[0], 1])
        theta_params = tf.reshape(theta_params, [theta_params.shape[0], 1])
        psi_params = tf.reshape(psi_params, [psi_params.shape[0], 1])
        return r_params, theta_params, psi_params
