import tensorflow as tf
import string
from models import SentenceAutoEncoder
from data_iterator import DataIterator
from data_preprocessor import DataPreprocessor

def load_vocabulary(vocabulary_file):
    # Returns a dictionary with {word, uid}
    vocab_dict = {}
    with open(vocabulary_file, 'r') as f:
        for line in f.readlines():
            word = line.rstrip()
            assert(word not in vocab_dict)

            vocab_dict[word] = len(vocab_dict)

    return vocab_dict

def main():
    # Step 0: Preprocess data
    dp = DataPreprocessor()

    # Step 1: Load Dataset
    data_iterator = DataIterator(dp.sentences_file, dp.vocabulary_file)

    # Step 2: Create Auto Encoder
    ranndy = SentenceAutoEncoder(data_iterator)

    # Step 3: Run training
    ranndy.train()
    # Step 4: Evaluate Results

    # Step 5: Try some encoding/decoding
    # ranndy.infer()

if __name__ == '__main__':
    main()