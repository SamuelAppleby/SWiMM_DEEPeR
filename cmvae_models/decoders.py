'''
file: decoders.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU

class ImgDecoder(Model):
    def __init__(self):
        super(ImgDecoder, self).__init__()
        self.create_model()

    def call(self, z):
        x1 = self.dense(z)
        x2 = self.reshape(x1)
        x3 = self.deconv1(x2)
        x4 = self.deconv2(x3)
        x5 = self.deconv3(x4)
        x6 = self.deconv4(x5)
        x7 = self.deconv5(x6)
        img_recon = self.deconv6(x7)
        return img_recon

    def create_model(self):

        print('[ImgDecoder] Creating layers')

        self.dense = Dense(units=1024, name='p_img_dense')
        self.reshape = Reshape((1, 1, 1024))

        # for 64x64 img
        self.deconv1 = Conv2DTranspose(filters=128, kernel_size=4, strides=1, padding='valid', activation='relu')
        self.deconv2 = Conv2DTranspose(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', dilation_rate=3)
        self.deconv3 = Conv2DTranspose(filters=64, kernel_size=6, strides=1, padding='valid', activation='relu', dilation_rate=2)
        self.deconv4 = Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu', dilation_rate=1)
        self.deconv5 = Conv2DTranspose(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', dilation_rate=1)
        self.deconv6 = Conv2DTranspose(filters=3, kernel_size=6, strides=1, padding='valid', activation='tanh')

        print('[ImgDecoder] Done with creating model')