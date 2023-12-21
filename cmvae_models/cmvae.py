import tensorflow as tf
from keras.backend import random_normal
from keras.layers import Lambda

from cmvae_models import dronet, decoders, transformer


# model definition class
class Cmvae(tf.keras.Model):
    def __init__(self, n_z, gate_dim=3, seed=None):
        super(Cmvae, self).__init__()
        # create the 3 base models:
        self.q_img = dronet.Dronet(num_outputs=n_z * 2, include_top=True)
        self.p_img = decoders.ImgDecoder()
        self.p_gate = decoders.GateDecoder(gate_dim=gate_dim)
        # Create sampler
        self.mean_params = Lambda(lambda x: x[:, : n_z])
        self.stddev_params = Lambda(lambda x: x[:, n_z:])
        self.seed = seed

    @tf.function
    def call(self, x, mode):
        # Possible modes for reconstruction:
        # 0: img -> img + gate
        # 1: img -> img
        # 2: img -> gate
        x = self.q_img(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5 * self.stddev_params(x))
        eps = random_normal(tf.shape(stddev), seed=self.seed)
        z = means + eps * stddev
        if mode == 0:
            img_recon = self.p_img(z)
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon, means, stddev, z
        elif mode == 1:
            img_recon = self.p_img(z)
            gate_recon = False
            return img_recon, gate_recon, means, stddev, z
        elif mode == 2:
            img_recon = False
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon, means, stddev, z

    def encode(self, x):
        x = self.q_img(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5 * self.stddev_params(x))
        eps = random_normal(tf.shape(stddev))
        z = means + eps * stddev
        return z, means, stddev

    def decode(self, z, mode):
        # Possible modes for reconstruction:
        # 0: z -> img + gate
        # 1: z -> img
        # 2: z -> gate
        if mode == 0:
            img_recon = self.p_img(z)
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon
        elif mode == 1:
            img_recon = self.p_img(z)
            gate_recon = False
            return img_recon, gate_recon
        elif mode == 2:
            img_recon = False
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon


# model definition class
class CmvaeDirect(tf.keras.Model):
    def __init__(self, n_z, seed=None):
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
        self.seed = seed

    @tf.function
    def call(self, x, mode):
        # Possible modes for reconstruction:
        # 0: img -> img + gate
        # 1: img -> img
        # 2: img -> gate
        z, means, stddev = self.encode(x)
        img_recon, gate_recon = self.decode(z, mode)
        return img_recon, gate_recon, means, stddev, z

    def encode(self, x):
        x = self.q_img(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5 * self.stddev_params(x))
        eps = tf.keras.backend.random_normal(tf.shape(stddev), seed=self.seed)
        z = means + eps * stddev
        return z, means, stddev

    def decode(self, z, mode):
        # Possible modes for reconstruction:
        # 0: z -> img + gate
        # 1: z -> img
        # 2: z -> gate
        if mode == 0:
            img_recon = self.p_img(z)
            r_params, theta_params, psi_params = self.extract_gate_params(z)
            gate_recon = tf.keras.layers.concatenate([self.p_R(r_params), self.p_Theta(theta_params), self.p_Psi(psi_params)], axis=1)
            return img_recon, gate_recon
        elif mode == 1:
            img_recon = self.p_img(z)
            gate_recon = False
            return img_recon, gate_recon
        elif mode == 2:
            img_recon = False
            r_params, theta_params, psi_params = self.extract_gate_params(z)
            gate_recon = tf.keras.layers.concatenate([self.p_R(r_params), self.p_Theta(theta_params), self.p_Psi(psi_params)], axis=1)
            return img_recon, gate_recon

    def extract_gate_params(self, z):
        # extract part of z vector
        r_params = self.R_params(z)
        theta_params = self.Theta_params(z)
        psi_params = self.Psi_params(z)
        # reshape variables
        r_params = tf.reshape(r_params, [r_params.shape[0], 1])
        theta_params = tf.reshape(theta_params, [theta_params.shape[0], 1])
        psi_params = tf.reshape(psi_params, [psi_params.shape[0], 1])
        return r_params, theta_params, psi_params
