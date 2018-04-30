import os
import string

class DataPreprocessor:
    def __init__(self):
        self.source_file = '../data/raw_sentences.txt'
        self.vocabulary_file = '../data/vocabulary.txt'
        self.sentences_file = '../data/sentences.txt'

        # Step 0: Parse sentence datafile
        #   For now we assume sentences are in the sibling directory data/sentences.txt
        self.parse_sentences(self.source_file, self.sentences_file, self.vocabulary_file)

    def parse_sentences(self, input_file, sentence_output_file, word_output_file):
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
                fout.write(word + '\n')

        print(f"Vocabulary is of size {len(word_set)}")

    def sentence_generator(self, filename):
        # Generator that reads file and yields each sentence
        # The end of a sentence is defined by a sequence of <./!/?><<space><uppercase>>/newline>
        file = open(filename, 'r')
        curr_sentence = ''
        curr_char = file.read(1)
        while curr_char:
            curr_sentence += curr_char
            if curr_char == ".":
                # Check cases where end of sentence is marked with a '.'
                if curr_sentence[-3:] != "Mr." and curr_sentence[-4] != "Mrs." and \
                                curr_sentence[-3:] != "Dr." and curr_sentence[-3] != "Ms." and \
                                curr_sentence[-3:] != "Jr." and curr_sentence[-3] != "Sr.":
                    # Check titles
                    next_char = file.read(1)
                    if next_char == "" or next_char == "\n":
                        # If sequence is a period followed by a new line or the end of the file, yield sentence
                        yield curr_sentence
                        curr_sentence = ""
                    elif next_char == " ":
                        next_char = file.read(1)
                        if next_char.isupper() or next_char == "":
                            # If sequence is a period, space, upper case, yield sentence
                            yield curr_sentence
                            curr_sentence = next_char
                        else:
                            curr_sentence += " " + next_char
                    else:
                        curr_sentence += next_char

            elif curr_char == "!":
                # Check cases where end of sentence is marked with a '!'
                next_char = file.read(1)
                if next_char == "" or next_char == "\n":
                    # If sequence is a period followed by a new line or the end of the file, yield sentence
                    yield curr_sentence
                    curr_sentence = ""
                elif next_char == " ":
                    next_char = file.read(1)
                    if next_char.isupper():
                        # If sequence is a period, space, upper case, yield sentence
                        yield curr_sentence
                        curr_sentence = next_char
                    else:
                        curr_sentence += " " + next_char
                else:
                    curr_sentence += next_char
            elif curr_char == "?":
                next_char = file.read(1)
                if next_char == "" or next_char == "\n":
                    # If sequence is a period followed by a new line or the end of the file, yield sentence
                    yield curr_sentence
                    curr_sentence = ""
                elif next_char == " ":
                    next_char = file.read(1)
                    if next_char.isupper():
                        # If sequence is a period, space, upper case, yield sentence
                        yield curr_sentence
                        curr_sentence = next_char
                    else:
                        curr_sentence += " " + next_char
                else:
                    curr_sentence += next_char

            curr_char = file.read(1)
            while curr_char == "\n":
                if len(curr_sentence) != 0 and curr_sentence[-1:] != " ":
                    curr_sentence += " "
                curr_char = file.read(1)
        file.close()

    def split_sentence(self, filename):
        # For each sentence in a file, write the sentence out on a new line
        placeholder_file = open("q5.out", "w")
        for sentence in self.sentence_generator(filename):
            placeholder_file.write(sentence + "\n")
        placeholder_file.close()