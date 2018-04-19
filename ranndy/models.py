import tensorflow as tf
import tensorflow.contrib.lookup as lookup

# Heavily based on: https://www.tensorflow.org/tutorials/seq2seq

'''
Various TODOs:
- Map rare known words to UNK (unknown) token to limit vocabulary size.
- Initialize embedding with CBOW or Skip-Gram.
'''

class SentenceAutoEncoder:
    """ Sentence auto encoder """

    def __init__(self, sentences, vocabulary):
        """"
        :param sentences: [tf.data.Dataset]
        :param vocabulary: [dict]
        """

        self.embedding_size = 512
        self.lstm_size = 512

        # Initialize dataset iterator
        self.iterator = sentences.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

        # Create vocabulary lookup
        self.vocab_size = len(vocabulary)
        self.table = lookup.HashTable(
            lookup.KeyValueTensorInitializer(list(vocabulary.keys()),
                                             list(vocabulary.values())), -1)

        # Setup embedding
        self.embedding = None
        self.embedding_output = None
        self._build_embedding(shape=[self.vocab_size, self.embedding_size])

    def _build_embedding(self, shape):

        self.embedding = tf.get_variable(
            "embedding", shape=shape, initializer=tf.random_normal_initializer)

        self.embedding_output = tf.nn.embedding_lookup(self.embedding, self.table.lookup(self.next_element))

    def _build_encoder(self):
        pass

    def _build_decoder(self):
        pass

    def train(self):
        pass

    def infer(self):
        with tf.Session() as sess:
            self.table.init.run()
            sess.run(tf.global_variables_initializer())
            for i in range(3):
                sentence, indexes, embeddings = sess.run([self.next_element,
                                                          self.table.lookup(self.next_element),
                                                          self.embedding_output])
                print("Sentence: ",[byte_str.decode('UTF-8') for byte_str in sentence])
                print("Indexes: ", indexes)
                print("Embeddings: ", embeddings)