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
        self.src = None
        self.tgt = None
        self.len = None

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
        # Add target (one-hot encoding) to dataset
        sentences = sentences.map(lambda src: (src, src))
        # Add length of each sentence to dataset
        sentences = sentences.map(lambda src, tgt: (src, tgt, tf.size(src)))

        # TODO: properly define and add eos token
        sentences = sentences.padded_batch(self.batch_size,
                   padded_shapes=(
                       tf.TensorShape([None]),  # src
                       tf.TensorShape([None]),  # tgt
                       tf.TensorShape([])),  # len
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
        self.src, self.tgt, self.len = self.next_element