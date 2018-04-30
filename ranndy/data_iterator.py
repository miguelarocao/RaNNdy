import tensorflow as tf
import string
import tensorflow.contrib.lookup as lookup

'''
Various TODOs:
- Map rare known words to UNK (unknown) token to limit vocabulary size.
- Initialize embedding with CBOW or Skip-Gram.
'''

class DataIterator:
    def __init__(self, data_file, vocab_file):
        """"
        :param sentences: [tf.data.Dataset]
        :param vocabulary: [dict]
        """
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.vocab = {}

        self.batch_size = 32

        # Dataset tensors
        self.source = None # Each datapoint is a sentence represented as a vector of integers (word labels)
        self.target = None # Each datapoint is a a sentence represented as a vector of integers (word labels)
        self.sentence_length = None # Each datapoint is an integer

        self.load_vocabulary()
        self.load_dataset()

    def load_vocabulary(self):
        # Returns a dictionary with {word, uid}
        with open(self.vocab_file, 'r') as f:
            for line in f.readlines():
                word = line.rstrip()
                assert (word not in self.vocab)

                self.vocab[word] = len(self.vocab)

        self.vocab_size = len(self.vocab)

        # Create vocabulary lookup
        self.table = lookup.HashTable(
            lookup.KeyValueTensorInitializer(list(self.vocab.keys()),
                                             list(self.vocab.values())), -1)

    def load_dataset(self):
        # Create dataset
        sentences = tf.data.TextLineDataset(self.data_file)

        # Split dataset sentences into vector of words
        sentences = sentences.map(lambda s: tf.string_split([s]).values)

        # Update dataset to use word keys instead of strings
        sentences = sentences.map(lambda words: (tf.cast(self.table.lookup(words), tf.int32)))

        # Add target labels to dataset
        sentences = sentences.map(lambda src: (src, src))

        # Add length of each sentence to dataset
        sentences = sentences.map(lambda src, tgt: (src, tgt, tf.size(src)))

        # Shuffle data on each iteration
        #   Note: On larger datasets we will probably have to increase the buffer size for appropriate randomness.
        sentences = sentences.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

        # TODO: properly define and add eos token
        sentences = sentences.padded_batch(self.batch_size,
                   padded_shapes=(
                       tf.TensorShape([None]),  # source
                       tf.TensorShape([None]),  # target
                       tf.TensorShape([])),  # sentence_length
                       # Pad the source and target sequences with eos tokens.
                       # (Though notice we don't generally need to do this since
                       # later on we will be masking out calculations past the true sequence.
                       # padding_values=(
                       #     src_eos_id,  # src
                       #     0) # src_len
                    )
        # Initialize dataset iterator
        self.iterator = sentences.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.source, self.target, self.sentence_length = self.next_element