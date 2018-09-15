# Based on: https://arxiv.org/pdf/1511.06349.pdf
# TODO: Word dropout and historyless decoding

from models.auto_encoder import SentenceAutoEncoder
import tensorflow as tf


class SentenceVAE(SentenceAutoEncoder):

    def __init__(self, data_iterator, mode=tf.estimator.ModeKeys.TRAIN):

        # Instance variables associated with VAE
        self.encoding_mean = None
        self.encoding_log_var = None # Note: Log(variance) is learned to ensure positivity
        self.sampled_state = None
        self.reconstruction_loss = None
        self.kl_loss = None
        self.kl_weight_scaling = 10 # TODO: Tune these values
        self.kl_weight_offset = 5000

        super().__init__(data_iterator, mode)

    def _build_encoder(self):
        _, encoder_state_tuple = self._build_encoder_rnn()

        # Since encoder state is an LSTMStateTuple, we extract only the state.
        # Documentation: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
        encoder_state = encoder_state_tuple.c

        # Layers to convert output to mean and variance
        #   Note: Paper uses linear layer (i.e. no activation)
        self.encoding_mean = tf.layers.dense(encoder_state, self.lstm_size, activation=None, name="linear_mean_layer")
        self.encoding_log_var = tf.layers.dense(encoder_state, self.lstm_size, activation=None, name="linear_var_layer")

        # Set decoder state input as random sample from generated decoder distribution
        self.sampled_state = self.gaussian_sample(self.encoding_mean, tf.exp(self.encoding_log_var))
        self.decoder_state_input = tf.contrib.rnn.LSTMStateTuple(c=self.sampled_state, h=encoder_state_tuple.h)

    def _build_trainer(self):
        # Build reconstruction loss
        self.reconstruction_loss = self._get_reconstruction_loss()

        # Build KL loss
        self.kl_loss = self.kl_divergence_multigauss(self.encoding_mean, tf.exp(self.encoding_log_var))/self.batch_size

        # Build KL loss weighting
        self.train_step_count = tf.get_variable("train_step_count", shape=[], initializer=tf.constant_initializer(-1))
        self.kl_weight = tf.sigmoid(tf.assign_add(self.train_step_count, 1)*self.kl_weight_scaling + self.kl_weight_offset)

        # Complete loss
        self.train_loss = self.reconstruction_loss + self.kl_weight*self.kl_loss

        # Optimization
        self.update_step = self._build_optimizer(self.train_loss)

    def gaussian_sample(self, mean, variance):
        eps = tf.random_normal(tf.shape(mean), mean=0, stddev=1.0)

        # mean + eps.*var (Where .* is element-wise multiplication)
        return tf.add(mean, tf.multiply(eps, variance))

    def kl_divergence_multigauss(self, mean, variance):
        # Source: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        return 0.5*tf.reduce_sum(tf.pow(variance, 2) + tf.pow(mean, 2) - tf.log(tf.pow(variance, 2)) - 1)
