import tensorflow as tf
import string
from models import SentenceAutoEncoder

def parse_sentences(input_file, sentence_output_file, word_output_file):
    # Removes punctuation from sentences and creates a vocabulary.

    printable = string.printable
    word_set = set()
    with open(input_file, 'r') as fin, open(sentence_output_file, 'w') as fout:
        for line in fin.readlines():
            # TODO: Regex this

            line = line.rstrip().lstrip()

            # Remove weird unicode characters
            line = ''.join(filter(lambda x: x in printable, line))

            if not line:
                continue
            sentence = line.translate(str.maketrans("", "", string.punctuation)).lower()
            words = sentence.split(' ')
            for word in words:
                if word == '':
                    continue
                word_set.add(word)
            fout.write(sentence + '\n')

    with open(word_output_file, 'w') as fout:
        for word in word_set:
            fout.write(word+ '\n')

    print(f"Vocabulary is of size {len(word_set)}")


def load_dataset(sentence_file):
    sentences = tf.data.TextLineDataset(sentence_file)

    # Split sentences into vector of words
    sentences = sentences.map(lambda s: tf.string_split([s]).values)

    return sentences

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
    source_file = '../data/raw_sentences.txt'
    vocabulary_file = '../data/vocabulary.txt'
    sentences_file = '../data/sentences.txt'

    # Step 0: Parse sentence datafile
    #   For now we assume sentences are in the sibling directory data/sentences.txt
    parse_sentences(source_file, sentence_output_file=sentences_file, word_output_file=vocabulary_file)

    # Step 1: Load sentences dataset and vocabulary
    sentences = load_dataset(sentences_file)
    vocab = load_vocabulary(vocabulary_file)

    # Step 2: Create Auto Encoder
    ranndy = SentenceAutoEncoder(sentences, vocab)

    # Step 3: Run training

    # Step 4: Evaluate Results

    # Step 5: Try some encoding/decoding
    ranndy.infer()

if __name__ == '__main__':
    main()