def byte_vec_to_sentence(vec, detokenizer):
    """
    Converts a vector of byte strings to a sentence.
    :param vec: [List] Vector of byte strings.
    :param detokenizer: NLTK detokenizer.
    :return: [String] Output sentence.
    """
    return detokenizer.detokenize(map(lambda x: x.decode("utf-8"), vec))
