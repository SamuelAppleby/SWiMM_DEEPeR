'''
file: cmvae.py
author: Kirsten Richardson
date: 2021
NB rolled back from TF2 to TF1, and three not four state variables

code taken from: https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation
author: Rogerio Bonatti et al
'''

import tensorflow as tf
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape
from cmvae_models import dronet
from cmvae_models import decoders
from cmvae_models import transformer

# model definition class
class CmvaeDirect(object):
    def __init__(self, n_z=10, state_dim=3, res=64, learning_rate=None, beta=None, trainable_model=True, big_data=False): 
        super(CmvaeDirect, self).__init__()
        # create the base models:
        self.q_img = dronet.Dronet(num_outputs=n_z*2)
        self.p_img = decoders.ImgDecoder()
        self.p_R = transformer.NonLinearTransformer()
        self.p_Theta = transformer.NonLinearTransformer()
        self.p_Psi = transformer.NonLinearTransformer()
        # Create sampler
        self.mean_params = Lambda(lambda x: x[:, : n_z])
        self.stddev_params = Lambda(lambda x: x[:, n_z:])
        self.R_params = Lambda(lambda x: x[:, 0])
        self.Theta_params = Lambda(lambda x: x[:, 1]) 
        self.Psi_params = Lambda(lambda x: x[:, 2])
        # initialize graph as instance variable
        self.graph = None
        # initialize input and output tensors as instance variables
        self.img_gt = None
        self.state_gt = None
        self.img_recon = None
        self.state_recon = None
        self.init_op = None
        self.img_data = None 
        self.state_data = None
        self.batch_size = None 
        # set z_size, learning rate, beta, res and state_dim
        self.z_size = n_z
        self.learning_rate = learning_rate
        self.beta = beta
        self.res = res
        self.state_dim = state_dim
        # flag whether training/testing model, or using as pretrained model in larger framework e.g. RL training/inference
        self.is_training = trainable_model
        # flag whether the img_data placeholder will be receiving images or filepaths to images
        self.big_data = big_data

        with tf.variable_scope('cmvae_direct', reuse=False):      
            self._build_graph()

        with self.graph.as_default():
            self.params = tf.trainable_variables()

        self._init_session()

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # build data pipeline

            # placeholders to receive data at beginning of training/testing loop each epoch
            if self.big_data:
                self.img_data = tf.placeholder(tf.string, shape=None)
            else:
                self.img_data = tf.placeholder(tf.float32, shape=[None, self.res, self.res, 3])
            self.state_data = tf.placeholder(tf.float64, shape=[None, self.state_dim])
            self.batch_size = tf.placeholder(tf.int64)

            # convert to tf format dataset and prepare batches
            if self.big_data:
                dataset = tf.data.Dataset.from_tensor_slices((self.img_data, self.state_data)).map(self.load_images).batch(self.batch_size)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((self.img_data, self.state_data)).batch(self.batch_size)
            
            # create iterator of the correct shape and type
            iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

            # create the initialisation operation
            self.init_op = iterator.make_initializer(dataset)

            # get the next image and state info to pass through model
            self.img_gt, self.state_gt = iterator.get_next()

            # model architecture                          

            # encoder
            x = self.q_img(self.img_gt)

            # VAE
            self.means = self.mean_params(x)
            self.stddev = tf.math.exp(0.5 * self.stddev_params(x))
            self.eps = random_normal(tf.shape(self.stddev))
            self.z = self.means + self.eps * self.stddev

            # decoders
            self.r_params, self.theta_params, self.psi_params = self.extract_state_params(self.z) 
            self.state_recon = tf.keras.layers.concatenate([self.p_R(self.r_params), self.p_Theta(self.theta_params), self.p_Psi(self.psi_params)], axis=1) 
            self.img_recon = self.p_img(self.z)

            # train / test ops
            if self.is_training:

                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                # calculate losses
                self.img_loss = tf.losses.mean_squared_error(self.img_gt, self.img_recon)
                self.state_loss = tf.losses.mean_squared_error(self.state_gt, self.state_recon)
                self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + self.stddev - tf.math.pow(self.means, 2) - tf.math.exp(self.stddev)), axis=1))

                # update running averages
                self.img_loss = tf.reduce_mean(self.img_loss)
                self.state_loss = tf.reduce_mean(self.state_loss)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                
                # combine loss components
                self.total_loss = self.img_loss + self.state_loss + self.beta*self.kl_loss

                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)            
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.total_loss)  
                self.train_op = self.optimizer.apply_gradients(
                    grads, global_step=self.global_step, name='train_step')

            # initialize vars
            self.init = tf.global_variables_initializer()           


    def _init_session(self):
        """Launch tensorflow session and initialize variables"""
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close tensorflow session """
        self.sess.close()

    def encode(self, x):
        """
        :param img_gt: (np.ndarray)
        :return: (np.ndarray), (np.ndarray), (np.ndarray)
        """
        return self.sess.run([self.means, self.stddev, self.z], feed_dict={self.img_gt: x})

    def decode(self, z):
        """
        :param z: (np.ndarray)
        :return: (np.ndarray), (np.ndarray)
        """
        return self.sess.run([self.img_recon, self.state_recon], feed_dict={self.z: z})

    def extract_state_params(self, z):

        # extract part of z vector
        r_params = self.R_params(z)
        theta_params = self.Theta_params(z)
        psi_params = self.Psi_params(z)

        # reshape variables
        r_params = tf.reshape(r_params, [tf.shape(r_params)[0], 1])
        theta_params = tf.reshape(theta_params, [tf.shape(theta_params)[0], 1])
        psi_params = tf.reshape(psi_params, [tf.shape(psi_params)[0], 1])

        return r_params, theta_params, psi_params

    def load_images(self, img, state):
        # read in image from filepath and decode
        image = tf.io.read_file(img)
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
        # cv2 used to write images so values in order BGR, switch first and last value so RGB
        # NOTE if using donkey_10k, donkey_50k, or donkey_300k, comment out line below as already in order RGB due to mistake in recorder.py 
        image = tf.reverse(image, axis=[-1])
        # resize to height and width training with
        image = tf.image.resize_images(image, (self.res, self.res))
        # normalize pixel values
        image = tf.to_float(image) * (2. / 255) - 1

        return image, state

    def save_weights(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, save_path)

    def load_weights(self, weights_path):
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, weights_path)
