import nltk
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
        self.embedding_size = 512
        self.lstm_size = 512
        self.batch_size = data_iterator.batch_size
        self.num_epochs = 15
        self.max_gradient_norm = 1.
        self.learning_rate = 0.0001

        # Data iterator
        self.iterator = data_iterator

        # Setup embedding
        self.embedding = None
        self.enc_embedding_output = None
        self.dec_embedding_output = None
        self._build_embedding(shape=[self.iterator.vocab_size, self.embedding_size])
        self._build_encoder()
        self._build_decoder()
        self.input_words = self.iterator.reverse_table.lookup(tf.to_int64(self.iterator.source))
        self.output_words = self.iterator.reverse_table.lookup(tf.to_int64(self.outputs.sample_id))
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self._build_trainer()
            # Training, validation, and testing lists to run
            self.run = {}
            self.run["val"] = [self.loss, self.iterator.source, self.outputs.sample_id, self.input_words, self.output_words]
            self.run["train"] = self.run["val"] + [self.update_step]
            self.run["test"] = list(self.run["val"])

        # Saver
        self.saver = tf.train.Saver()
        self.checkpoint_path = "../data/checkpoints/checkpoint.ckpt"

        # Detokenizer
        self.detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()

    def _build_embedding(self, shape):
        self.embedding = tf.get_variable(
            "embedding", shape=shape, initializer=tf.random_normal_initializer)

        self.enc_embedding_output = tf.nn.embedding_lookup(self.embedding, self.iterator.source)
        self.dec_embedding_output = tf.nn.embedding_lookup(self.embedding, self.iterator.target_input)

    def _build_encoder(self):
        # Build RNN cell
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(self.encoder_cell, self.enc_embedding_output,
                                                                     sequence_length=self.iterator.source_length,
                                                                     dtype=tf.float32)

    def _build_decoder(self):
        # Projection layer: Necessary because LSTM output size (which is same as state size) will probably be different
        #   than vocabulary size.
        projection_layer = tf.layers.Dense(self.iterator.vocab_size, use_bias=False)

        # Build RNN cell
        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)

        # Helper
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.TrainingHelper(self.dec_embedding_output, self.iterator.target_length)
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding,
                                                              tf.fill([self.batch_size], 
                                                              tf.cast(self.iterator.sos_index, tf.int32)),
                                                              tf.cast(self.iterator.sos_index, tf.int32))

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
                tf.reduce_max(self.iterator.target_length) * 2))
            self.logits = self.outputs.rnn_output

    def _build_trainer(self):
        # TODO: Figure out why target_weights is necessary
        target_weights = tf.sequence_mask(self.iterator.target_length, dtype=self.logits.dtype)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iterator.target_output,
                                                                  logits=self.logits)

        self.loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    def run_epoch(self, sess, data_type, verbose=True):
        results = sess.run(self.run[data_type])

        in_words = [list(map(lambda x: x.decode(), y)) for y in results[3]]
        out_words = [list(map(lambda x: x.decode(), y)) for y in results[4]]

        # Calculate average bleu score over data
        cum_bleu_score = 0.
        for in_, out_ in zip(results[3], results[4]):
            cum_bleu_score += nltk.translate.bleu_score.sentence_bleu([in_], out_)
        avg_blue_score = cum_bleu_score / float(len(in_words))

        if verbose:
            print(f"Input tokens (size: {len(results[1][0])}): {results[1][0]}")
            print(f"Output tokens (size: {len(results[2][0])}): {results[2][0]}")
            print(f"Input words (size: {len(results[3][0])}): {' '.join(in_words[0])}")
            print(f"Output words (size: {len(results[4][0])}): {' '.join(out_words[0])}")

        return results[0], avg_blue_score


    def train(self, plot=False, verbose=True):
        # Not sure this is necessary
        assert (self.mode == tf.estimator.ModeKeys.TRAIN)
        losses = []
        bleu_scores = []
        data_type = "train"
        loss, bleu_score = None, None
        with tf.Session() as sess:
            self.iterator.table.init.run()
            self.iterator.reverse_table.init.run()
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer[data_type])

            start_time = time.time()

            epoch = 1
            first_batch = True
            while epoch <= self.num_epochs:
                ### Run a train step ###
                try:
                    loss, bleu_score = self.run_epoch(sess, data_type, verbose=verbose and first_batch)
                    losses.append(loss)
                    bleu_scores.append(bleu_score)
                    first_batch = False
                except tf.errors.OutOfRangeError:
                    # Finished going through the training dataset.  Go to next epoch.
                    print(f"[Epoch: {epoch}] {data_type.capitalize()} Loss: {loss} BLEU score: {bleu_score}")
                    loss, bleu_score = None, None
                    if data_type == "train":
                        data_type = "val"
                    else:
                        data_type = "train"
                        epoch += 1
                    print(data_type)
                    sess.run(self.iterator.initializer[data_type])
                    first_batch = True
                    continue

            print(f"Run time: {time.time() - start_time}")

            self.saver.save(sess, self.checkpoint_path)
            print(f"Model saved to {self.checkpoint_path}")

        if plot:
            plt.plot(losses)
            plt.ylabel('crossent error')
            plt.xlabel('batch #')
            plt.show()

    def test(self, verbose=True):
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            self.iterator.table.init.run()
            self.iterator.reverse_table.init.run()
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer["test"])

            start_time = time.time()

            ### Run a test step ###
            while True:
                try:
                    loss, bleu_score = self.run_epoch(sess, "test", verbose=verbose)
                    print(f"Test results\nLoss: {loss} BLEU score: {bleu_score}")
                except tf.errors.OutOfRangeError:
                    # Finished going through the training dataset.  Go to next epoch.
                    break

            print(f"Run time: {time.time() - start_time}")

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
                originals, results = sess.run([self.input_words, self.output_words])
                for original, result in zip(originals, results):
                    print(f"Original: {dh.byte_vec_to_sentence(original, self.detokenizer)} Result: {dh.byte_vec_to_sentence(result, self.detokenizer)}")
