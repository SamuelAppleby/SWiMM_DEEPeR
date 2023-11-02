"""
file: transformer.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al.
"""

import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()


class NonLinearTransformer(tf.keras.Model):
    def __init__(self):
        super(NonLinearTransformer, self).__init__()
        self.create_model()

    def call(self, x):
        x1 = self.dense0(x)
        x2 = self.dense1(x1)
        x3 = self.dense2(x2)
        return x3

    def create_model(self):
        print('[NonLinearTransformer] Creating layers')

        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1, activation='linear')

        print('[NonLinearTransformer] Done with creating model')
