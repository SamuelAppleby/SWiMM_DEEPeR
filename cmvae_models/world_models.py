import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()


class WMEncoder(tf.keras.Model):
    def __init__(self, num_outputs):
        super(WMEncoder, self).__init__()
        self.create_model(num_outputs)

    def call(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.reshape(x4)
        mu = self.dense1(x5)
        logvar = self.dense2(x5)
        return mu, logvar

    def create_model(self, num_outputs):
        print('[WMEncoder] Creating layers')

        self.dense1 = tf.keras.layers.Dense(units=num_outputs, name="enc_fc_mu")
        self.dense2 = tf.keras.layers.Dense(units=num_outputs, name="enc_fc_log_var")
        self.reshape = tf.keras.layers.Reshape((-1, 2 * 2 * 256))

        # for 64x64 img
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, activation='relu')

        print('[WMEncoder] Done with creating model')


# class WMDecoder(Model):
#     def __init__(self):
#         super(WMDecoder, self).__init__()
#         self.create_model()

#     def call(self, z):
#         x1 = self.dense(z)
#         x2 = self.reshape(x1)
#         x3 = self.deconv1(x2)
#         x4 = self.deconv2(x3)
#         x5 = self.deconv3(x4)
#         img_recon = self.deconv4(x5)
#         return img_recon

#     def create_model(self):

#         print('[WMDecoder] Creating layers')

#         self.dense = Dense(units=4*256, name='dec_fc')
#         self.reshape = Reshape((1, 1, 4*256))

#         # for 64x64 img
#         self.deconv1 = Conv2DTranspose(filters=128, kernel_size=5, strides=2, activation='relu')
#         self.deconv2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation='relu')
#         self.deconv3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation='relu')
#         self.deconv4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation='sigmoid')

#         print('[WMDecoder] Done with creating model')


class WMDecoder(tf.keras.Model):
    def __init__(self):
        super(WMDecoder, self).__init__()
        self.create_model()

    def call(self, z):
        x1 = self.dense(z)
        x2 = self.reshape(x1)
        x3 = self.deconv1(x2)
        x4 = self.deconv2(x3)
        x5 = self.deconv3(x4)
        img_recon = self.deconv4(x5)
        # x7 = self.deconv5(x6)
        # img_recon = self.deconv6(x7)
        return img_recon

    def create_model(self):
        print('[WMDecoder] Creating layers')

        self.dense = tf.keras.layers.Dense(units=1024, name='p_img_dense')
        self.reshape = tf.keras.layers.Reshape((1, 1, 1024))

        # for 64x64 img
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation='relu')
        self.deconv4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation='sigmoid')
        # self.deconv5 = Conv2DTranspose(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu', dilation_rate=1)
        # self.deconv6 = Conv2DTranspose(filters=3, kernel_size=6, strides=1, padding='valid', activation='tanh')

        print('[WMDecoder] Done with creating model')
