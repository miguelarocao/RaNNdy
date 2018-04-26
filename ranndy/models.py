import matplotlib.pyplot as plt
import tensorflow as tf
import time

# Heavily based on: https://www.tensorflow.org/tutorials/seq2seq

class SentenceAutoEncoder:
    """ Sentence auto encoder """

    def __init__(self, data_iterator):
        """
        :param data_iterator: iterate through data
        """

        self.embedding_size = 512
        self.lstm_size = 512
        # TODO: set up hyperparams properly
        self.batch_size = data_iterator.batch_size
        self.num_epochs = 10
        self.max_gradient_norm = 1.
        self.learning_rate = 0.0001
        # Roughly based on 2 std from https://english.stackexchange.com/questions/276715/standard-deviation-for-average-sentence-and-paragraph-length
        self.max_time = 50

        # Data iterator
        self.iterator = data_iterator

        # Setup embedding
        self.embedding = None
        self.embedding_output = None
        self._build_embedding(shape=[self.iterator.vocab_size, self.embedding_size])
        self._build_encoder()
        self._build_decoder()
        self._build_trainer()

    def _build_embedding(self, shape):
        self.embedding = tf.get_variable(
            "embedding", shape=shape, initializer=tf.random_normal_initializer)

        self.embedding_output = tf.nn.embedding_lookup(self.embedding, self.iterator.src)

    def _build_encoder(self):
        # Build RNN cell
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.embedding_output,
                                                                     sequence_length=self.iterator.len,
                                                                     dtype=tf.float32)

    def _build_decoder(self):
        # TODO: Unsure what this is used for
        projection_layer = tf.layers.Dense(self.iterator.vocab_size, use_bias=False)
        # Build RNN cell
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(self.embedding_output, self.iterator.len)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        self.outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.logits = self.outputs.rnn_output

    def _build_trainer(self):
        target_weights = tf.sequence_mask(self.iterator.len, dtype=self.logits.dtype)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iterator.tgt,
                                                                  logits=self.logits)
        self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    def train(self):
        losses = []
        with tf.Session() as sess:
            self.iterator.table.init.run()
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())

            start_train_time = time.time()

            epoch = 1
            while epoch <= self.num_epochs:
                ### Run a step ###
                start_time = time.time()
                try:
                    loss, _ = sess.run([self.train_loss, self.update_step])
                    losses.append(loss)

                except tf.errors.OutOfRangeError:
                    # Finished going through the training dataset.  Go to next epoch.
                    print("# Finished an epoch %d." % epoch)
                    epoch += 1
                    sess.run(self.iterator.initializer)

                    continue

        plt.plot(losses)
        plt.ylabel('loss')
        plt.xlabel('steps')
        plt.show()

    def infer(self):
        with tf.Session() as sess:
            self.table.init.run()
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            for i in range(3):
                sentence, length, embeddings = sess.run([self.src,
                                                         self.len,
                                                         self.embedding_output])
                print("Sentence: ", sentence)
                print("Length: ", length)
                print("Embeddings: ", embeddings)