import matplotlib.pyplot as plt
import tensorflow as tf
import time
import data_helpers as dh


# Heavily based on: https://www.tensorflow.org/tutorials/seq2seq

class SentenceAutoEncoder:
    """ Sentence auto encoder """

    def __init__(self, data_iterator, mode=tf.estimator.ModeKeys.TRAIN):
        """
        :param data_iterator: [ranndy.DataIterator] iterate through data
        """

        # Train or Infer
        self.mode = mode

        # TODO: set up hyperparams properly
        self.embedding_size = 256
        self.lstm_size = 512
        self.batch_size = data_iterator.batch_size
        self.num_epochs = 10
        self.max_gradient_norm = 1.
        self.learning_rate = 0.0001

        # Data iterator
        self.iterator = data_iterator

        # Setup embedding
        self.embedding = None
        self.embedding_output = None
        self._build_embedding(shape=[self.iterator.vocab_size, self.embedding_size])
        self._build_encoder()
        self._build_decoder()
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_trainer()

        # Saver
        self.saver = tf.train.Saver()
        self.checkpoint_path = "../data/checkpoints/checkpoint.ckpt"

    def _build_embedding(self, shape):
        self.embedding = tf.get_variable(
            "embedding", shape=shape, initializer=tf.random_normal_initializer)

        self.embedding_output = tf.nn.embedding_lookup(self.embedding, self.iterator.source)

    def _build_encoder(self):
        # Build RNN cell
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.embedding_output,
                                                                     sequence_length=self.iterator.sentence_length,
                                                                     dtype=tf.float32)

    def _build_decoder(self):
        # Projection layer: Necessary because LSTM output size (which is same as state size) will probably be different
        #   than vocabulary size.
        projection_layer = tf.layers.Dense(self.iterator.vocab_size, use_bias=False)

        # Build RNN cell
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)

        # Helper
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.TrainingHelper(self.embedding_output, self.iterator.sentence_length)
        else:
            sos_token = self.iterator.lookup_indexes(tf.convert_to_tensor(self.iterator.sos_marker))
            eos_token = self.iterator.lookup_indexes(tf.convert_to_tensor(self.iterator.eos_marker))
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding,
                                                              tf.fill([self.batch_size], sos_token),
                                                              eos_token)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.encoder_state,
                                                  output_layer=projection_layer)

        # Dynamic decoding
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # Training
            self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            self.logits = self.outputs.rnn_output
        else:
            # Inference
            self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.round(
                tf.reduce_max(self.iterator.sentence_length) * 2))

    def _build_trainer(self):
        # TODO: Figure out why target_weights is necessary
        target_weights = tf.sequence_mask(self.iterator.sentence_length, dtype=self.logits.dtype)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iterator.target,
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
        assert (self.mode == tf.estimator.ModeKeys.TRAIN)
        losses = []
        with tf.Session() as sess:
            self.iterator.table.init.run()
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())

            start_time = time.time()

            epoch = 1
            while epoch <= self.num_epochs:
                ### Run a step ###
                try:
                    loss, _ = sess.run([self.train_loss, self.update_step])
                    losses.append(loss)

                except tf.errors.OutOfRangeError:
                    # Finished going through the training dataset.  Go to next epoch.
                    print(f"[Epoch: {epoch}] Train Loss: {loss}")
                    epoch += 1
                    sess.run(self.iterator.initializer)

                    continue

            print(f"Run time: {time.time() - start_time}")

            self.saver.save(sess, self.checkpoint_path)
            print(f"Model saved to {self.checkpoint_path}")

        plt.plot(losses)
        plt.ylabel('crossent error')
        plt.xlabel('batch #')
        plt.show()

    def infer(self, num_batch_infer):
        assert (self.mode == tf.estimator.ModeKeys.PREDICT)
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            print(f"Model loaded from {self.checkpoint_path}")
            self.iterator.table.init.run()
            self.iterator.reverse_table.init.run()
            sess.run(self.iterator.initializer)
            for i in range(num_batch_infer):
                #self.iterator.lookup_words(self.outputs.sample_id)
                originals, results = sess.run([self.iterator.lookup_words(self.iterator.source),
                                               self.iterator.lookup_words(self.outputs.sample_id)])
                for original, result in zip(originals, results):
                    print(f"Original: {dh.byte_vec_to_sentence(original)} Result: {dh.byte_vec_to_sentence(result)}")
