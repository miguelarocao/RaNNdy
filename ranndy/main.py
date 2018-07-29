import argparse

import tensorflow as tf

from data_iterator import DataIterator
from models import SentenceAutoEncoder

def main():
    # Define argument parsing
    parser = argparse.ArgumentParser(description="Tell RaNNdy what to do.")
    parser.add_argument('--mode', help='Whether to train or infer.', default='train', choices=['train', 'infer'])
    parser.add_argument('--sentence_tokens', help ='Tokenized sentences to train on.', default='../data/sentence_tokens.csv')
    parser.add_argument('--vocab', help ='Vocabulary to use for training.', default='../data/vocabulary.csv')
    args = parser.parse_args()

    if args.mode == 'train':
        # Step 1: Load Dataset
        data_iterator = DataIterator(args.sentence_tokens, args.vocab)

        # Step 2: Create Auto Encoder in Trainng Mode
        ranndy = SentenceAutoEncoder(data_iterator, tf.estimator.ModeKeys.TRAIN)

        # Step 3: Train
        ranndy.train()
    else:
        # Step 1: Load Dataset w/ batch size of 1
        data_iterator = DataIterator(args.sentence_tokens, args.vocab, batch_size=2, shuffle=False)

        # Step 2: Create Auto Encoder in Inference Mode
        ranndy = SentenceAutoEncoder(data_iterator, tf.estimator.ModeKeys.PREDICT)

        # Step 3: Infer
        ranndy.infer(num_batch_infer=1)

if __name__ == '__main__':
    main()