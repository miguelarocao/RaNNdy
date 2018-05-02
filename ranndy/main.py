from models import SentenceAutoEncoder
from data_iterator import DataIterator
from data_preprocessor import DataPreprocessor
import tensorflow as tf
import argparse

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
    # Define argument parsing
    parser = argparse.ArgumentParser(description="Tell RaNNdy what to do.")
    parser.add_argument('--mode', help='Whether to train or infer.', default='train', choices=['train', 'infer'])
    args = parser.parse_args()

    # Step 0: Preprocess data
    dp = DataPreprocessor()

    if args.mode == 'train':
        # Step 1: Load Dataset
        data_iterator = DataIterator(dp.sentences_file, dp.vocabulary_file)

        # Step 2: Create Auto Encoder in Trainng Mode
        ranndy = SentenceAutoEncoder(data_iterator, tf.estimator.ModeKeys.TRAIN)

        # Step 3: Train
        ranndy.train()
    else:
        # Step 1: Load Dataset w/ batch size of 1
        data_iterator = DataIterator(dp.sentences_file, dp.vocabulary_file, batch_size=3, shuffle=False)

        # Step 2: Create Auto Encoder in Inference Mode
        ranndy = SentenceAutoEncoder(data_iterator, tf.estimator.ModeKeys.PREDICT)

        # Step 3: Infer
        ranndy.infer(num_batch_infer=1)

if __name__ == '__main__':
    main()