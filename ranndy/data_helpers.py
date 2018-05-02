def byte_vec_to_sentence(vec):
    # Converts a vector of byte strings to a sentence (single space separated string)
    return ' '.join(map(lambda x: x.decode("utf-8"), vec))