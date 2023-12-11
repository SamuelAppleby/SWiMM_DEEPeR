import tensorflow as tf


class ImgDecoder(tf.keras.Model):
    def __init__(self):
        super(ImgDecoder, self).__init__()
        self.create_model()

    @tf.function
    def call(self, z):
        return self.network(z)

    def create_model(self):
        print('[ImgDecoder] Starting create_model')
        dense = tf.keras.layers.Dense(units=1024, name='p_img_dense')
        reshape = tf.keras.layers.Reshape((1, 1, 1024))

        # for 64x64 img
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=1, padding='valid', activation='relu')
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu', dilation_rate=3)
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=6, strides=1, padding='valid', activation='relu', dilation_rate=2)
        deconv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu', dilation_rate=1)
        deconv5 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', dilation_rate=1)
        # deconv6 = Conv2DTranspose(filters=8, kernel_size=6, strides=2, padding='valid', activation='relu')
        deconv7 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=1, padding='valid', activation='tanh')
        self.network = tf.keras.Sequential([
            dense,
            reshape,
            deconv1,
            deconv2,
            deconv3,
            deconv4,
            deconv5,
            deconv7],
            name='p_img')

        print('[ImgDecoder] Done with create_model')


class GateDecoder(tf.keras.Model):
    def __init__(self, gate_dim):
        super(GateDecoder, self).__init__()
        self.create_model(gate_dim)

    @tf.function
    def call(self, z):
        return self.network(z)

    def create_model(self, gate_dim):
        print('[GateDecoder] Starting create_model')
        dense0 = tf.keras.layers.Dense(units=512, activation='relu')
        dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        dense3 = tf.keras.layers.Dense(units=16, activation='relu')
        dense4 = tf.keras.layers.Dense(units=gate_dim, activation='linear')
        self.network = tf.keras.Sequential([
            # dense0,
            # dense1,
            # dense2,
            # dense3,
            dense4],
            name='p_gate')

        print('[GateDecoder] Done with create_model')
