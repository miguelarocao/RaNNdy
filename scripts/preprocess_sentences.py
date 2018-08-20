import string
import argparse
import nltk
import csv
import re

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def parse_sentences(input_file, token_output_file, vocab_output_file, max_length=32, num_word_print=10):
    """
    Parses input sentences. Outputs tokenized sentences and a vocabulary with word counts.
    Note that all words are taken into account in the vocabulary, even if they are truncated due to the sentence being
    too long.

    :param input_file: [String] Input file containing sentences to parse. One sentence per line.
    :param token_output_file: [String] Output file or tokenized sentences.
    :param vocab_output_file: [String] Output file for vocabulary.
    :param max_length: [String] Maximum sentence length (based on word count) which will be stored  . Longer sentences are
        truncated.
    :param num_word_print: [Int] Number of words to print from the vocabulary. Ordered by frequency.
    """
    printable = string.printable
    word_dict = {} # {word: count}
    with open(input_file, 'r') as fin, open(token_output_file, 'w') as fout:
        writer = csv.writer(fout)
        for line in fin.readlines():

            line = line.rstrip().lstrip()

            # Remove weird unicode characters
            line = ''.join(filter(lambda x: x in printable, line))

            if not line:
                continue

            sentence = rap_word_expansion(line.lower())

            words = nltk.word_tokenize(sentence)

            # Remove punctuation
            words = [w for w in words if w not in string.punctuation]

            for word in words:
                if word not in word_dict:
                    word_dict[word] = 0
                word_dict[word] += 1

            writer.writerow(words[:max_length])

    with open(vocab_output_file, 'w') as fout:
        writer = csv.writer(fout)
        for word, count in word_dict.items():
            writer.writerow([word, count])

    # Print some info
    print(f"Vocabulary is of size {len(word_dict)}")
    sorted_word_list = [(word, count) for word, count in word_dict.items()]
    sorted_word_list.sort(key=lambda x: x[1], reverse=True)
    print(f"The {num_word_print} most common words are: {sorted_word_list[:num_word_print]}")
    print(f"The {num_word_print} least common words are: {sorted_word_list[-num_word_print:]}")

def rap_word_expansion(sentence):
    """
    Expands the contraction -in' to -ing in the input sentence. For example: rappin' becomes rapping.
    Only matches at the end of a word (requires space after).
    :param sentence: [String] Sentence to expand.
    :return: Expanded sentence.
    """
    sentence = expand_contractions(sentence)
    return sentence.replace("in' ", "ing ")

def expand_contractions(text, c_re=c_re):
    """ Expands common english contractions using dictionary at the top of the file """
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())

def main():
    parser = argparse.ArgumentParser(description="Preprocessing.")
    parser.add_argument('--input', help='Input file. Should be list of sentences.', default='../data/lyrics.txt')
    parser.add_argument('--token_output', help='Tokenized sentence output file.', default='../data/sentence_tokens.csv')
    parser.add_argument('--vocab_output', help='Vocabulary output file.', default='../data/vocabulary.csv')
    args = parser.parse_args()

    parse_sentences(args.input, args.token_output, args.vocab_output)


if __name__ == '__main__':
    main()