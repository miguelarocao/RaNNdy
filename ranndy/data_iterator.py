import tensorflow as tf
import string
import tensorflow.contrib.lookup as lookup

'''
Various TODOs:
- Map rare known words to UNK (unknown) token to limit vocabulary size.
- Initialize embedding with CBOW or Skip-Gram.
'''


class DataIterator:
    def __init__(self, data_file, vocab_file, batch_size=32, shuffle=True):
        """
        :param sentence_tokens: [tf.data.Dataset]
        :param vocabulary: [dict]
        """
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.vocab = []

        self.batch_size = batch_size
        self.sos_marker = "<s>"  # Start Of Sentence marker
        self.eos_marker = "<\s>"  # End Of Sentence marker

        # Dataset tensors
        self.source = None  # Each datapoint is a sentence represented as a vector of integers (word labels)
        self.target_input = None  # Each datapoint is a a sentence represented as a vector of integers (word labels)
        self.target_output = None  # Each datapoint is a a sentence represented as a vector of integers (word labels)
        self.source_length = None  # Each datapoint is an integer
        self.target_length = None  # Each datapoint is an integer

        self.load_vocabulary()
        self.load_dataset(shuffle)

    def load_vocabulary(self):
        # Add SOS and EOS tokens
        self.vocab += [self.sos_marker, self.eos_marker]

        # Add words from vocabulary file
        with open(self.vocab_file, 'r') as f:
            for line in f.readlines():
                word = line.split(',')[0]
                assert (word not in self.vocab)

                self.vocab.append(word)

        self.vocab_size = len(self.vocab)

        # Create vocabulary lookup
        self.table = tf.contrib.lookup.index_table_from_tensor(tf.constant(self.vocab), default_value=-1)

        # Create reverse lookup table
        self.reverse_table = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant(self.vocab), default_value="UNKNOWN")

    def load_dataset(self, shuffle):
        # Get start and end of sentence indices
        sos_index = self.lookup_indexes(self.sos_marker)
        eos_index = self.lookup_indexes(self.eos_marker)

        # Create dataset
        sentences = tf.data.TextLineDataset(self.data_file)

        # Split dataset sentences into vector of words
        sentences = sentences.map(lambda s: tf.string_split([s], delimiter=",").values)

        # Update dataset to use word keys instead of strings
        sentences = sentences.map(lambda words: self.lookup_indexes(words))

        # Add target inputs and outputs, concatenate <sos> to target_input start and <eos> to target_output end
        sentences = sentences.map(
            lambda src: (src, tf.concat(([sos_index], src), axis=0), tf.concat((src, [eos_index]), axis=0)))

        # Add length of each sentence to dataset
        sentences = sentences.map(lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_out)))

        # Shuffle data on each iteration
        #   Note: On larger datasets we will probably have to increase the buffer size for appropriate randomness.
        if shuffle:
            sentences = sentences.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

        sentences = sentences.padded_batch(self.batch_size,
                                           padded_shapes=(
                                               tf.TensorShape([None]),  # source
                                               tf.TensorShape([None]),  # target_input
                                               tf.TensorShape([None]),  # target_output
                                               tf.TensorShape([]),  # source_length
                                               tf.TensorShape([])),  # target_length
                                           # Pad the source and target sequences with eos tokens.
                                           # (Though notice we don't generally need to do this since
                                           # later on we will be masking out calculations past the true sequence.
                                           padding_values=(
                                               self.lookup_indexes(self.eos_marker),  # source
                                               self.lookup_indexes(self.eos_marker),  # target_input
                                               self.lookup_indexes(self.eos_marker),  # target_output
                                               0,  # source_length
                                               0)  # target_length
                                           )
        # Initialize dataset iterator
        self.iterator = sentences.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.source, self.target_input, self.target_output, self.source_length, self.target_length = self.next_element

    def lookup_indexes(self, words):
        """ Returns the index(es) for the given word(s) in the vocabulary """
        return tf.to_int32(self.table.lookup(tf.convert_to_tensor(words)))

    def lookup_words(self, indexes):
        """ Returns the vocabulary word(s) for the given index(es)"""
        return self.reverse_table.lookup(tf.to_int64(indexes))
