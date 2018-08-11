import tensorflow as tf
import csv
import heapq as hq
import math

class DataIterator:
    def __init__(self, data_file, vocab_file, batch_size=32, shuffle=True, max_vocab_size=-1):
        """
        :param data_file: [String] CSV file of sentence tokens.
        :param vocab_file: [String] CSV file of vocabulary with frequency.
        :param batch_size: [Int] Batch size for iteration.
        :param shuffle: [Boolean] Whether the data should be shuffled (by line).
        :param max_vocab_size: [Int] Maximum vocabulary size. -1 means no limit.
        """
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.vocab = []
        self.vocab_size = 0

        self.batch_size = batch_size
        self.sos_token = "<s>"  # Start Of Sentence token
        self.eos_token = "<\s>"  # End Of Sentence token
        self.unk_token = "<UNK>" # Unknown word token
        self.num_oov_buckets = 1 # Number of out of vocabulary buckets

        # Dataset tensors
        self.source = None  # Each datapoint is a sentence represented as a vector of integers (word labels)
        self.target_input = None  # Each datapoint is a a sentence represented as a vector of integers (word labels)
        self.target_output = None  # Each datapoint is a a sentence represented as a vector of integers (word labels)
        self.source_length = None  # Each datapoint is an integer
        self.target_length = None  # Each datapoint is an integer

        self.load_vocabulary(max_vocab_size)
        self.load_dataset(shuffle)

    def load_vocabulary(self, max_vocab_size=-1):
        """
        Loads the vocabulary and creates lookup tables.
        :param max_vocab_size: [Int] Maximum vocabulary size. -1 means no limit.
        :return:
        """
        # Due to out of vocabulary bucket, the working max_vocab_size is less
        working_max_vocab_size = max_vocab_size - self.num_oov_buckets

        # We'll use a heap to keep track of of our "max_vocab_size" most frequent vocabulary words.
        # For consistency, we break word frequency ties by lexicographical order
        vocab_heap = []

        # Add SOS and EOS tokens (with infinite frequency so they cannot be removed)
        hq.heappush(vocab_heap, (math.inf, self.sos_token))
        hq.heappush(vocab_heap, (math.inf, self.eos_token))

        # Add words from vocabulary file
        with open(self.vocab_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                heap_entry = (int(line[1]), line[0]) # Order by frequency, then word
                if len(vocab_heap) < working_max_vocab_size or max_vocab_size == -1:
                    hq.heappush(vocab_heap, heap_entry)
                else:
                    hq.heappushpop(vocab_heap, heap_entry)

        # Now convert our heap to the vocab
        self.vocab = set([x[1] for x in vocab_heap])
        self.vocab_size = len(self.vocab) + self.num_oov_buckets

        # Create vocabulary lookup. There is 1 OOV bucket for words outside the vocabulary.
        self.table = tf.contrib.lookup.index_table_from_tensor(tf.constant(list(self.vocab)), num_oov_buckets=self.num_oov_buckets)
        # Get start and end of sentence indices
        self.sos_index = self.table.lookup(tf.convert_to_tensor(self.sos_token))
        self.eos_index = self.table.lookup(tf.convert_to_tensor(self.eos_token))

        # Create reverse lookup table
        self.reverse_table = tf.contrib.lookup.index_to_string_table_from_tensor(
                                tf.constant(list(self.vocab)), default_value=self.unk_token)

    def load_dataset(self, shuffle):
        # Create dataset
        sentences = tf.data.TextLineDataset(self.data_file)

        # Split dataset sentences into vector of words
        sentences = sentences.map(lambda s: tf.string_split([s], delimiter=",").values)

        # Update dataset to use word keys instead of strings
        sentences = sentences.map(lambda words: self.table.lookup(words))

        # Add target inputs and outputs, concatenate <sos> to target_input start and <eos> to target_output end
        sentences = sentences.map(
            lambda src: (src, tf.concat(([self.sos_index], src), axis=0), tf.concat((src, [self.eos_index]), axis=0)))

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
                                               self.eos_index,  # source
                                               self.eos_index,  # target_input
                                               self.eos_index,  # target_output
                                               0,  # source_length
                                               0)  # target_length
                                           )
        # Initialize dataset iterator
        self.iterator = sentences.make_initializable_iterator()
        self.next_element = self.iterator.get_next()
        self.initializer = self.iterator.initializer
        self.source, self.target_input, self.target_output, self.source_length, self.target_length = self.next_element
