import string
import argparse
import nltk
import csv
import contraction_list as cl


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
            sentence = expand_contractions(sentence)

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
    return sentence.replace("in' ", "ing ")

def expand_contractions(sentence):
    """
    Expands common english contractions defined in the contraction_list.py file.
    cl.regex_form is a regex object that is searched and each occurance of the contraction is 
    replaced with its uncontracted counterpart.
    :param sentence: [String] Sentence to expand.
    :return: Expanded sentence.
    """
    def replace(match):
        return cl.contraction_dict[match.group(0)]
    return cl.compiled_regex.sub(replace, sentence)

def main():
    parser = argparse.ArgumentParser(description="Preprocessing.")
    parser.add_argument('--input', help='Input file. Should be list of sentences.', default='../data/lyrics.txt')
    parser.add_argument('--token_output', help='Tokenized sentence output file.', default='../data/sentence_tokens.csv')
    parser.add_argument('--vocab_output', help='Vocabulary output file.', default='../data/vocabulary.csv')
    args = parser.parse_args()

    parse_sentences(args.input, args.token_output, args.vocab_output)


if __name__ == '__main__':
    main()