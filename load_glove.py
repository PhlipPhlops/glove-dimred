import numpy as np
from scipy import spatial

DIMENSION = 50
print("Loading GloVe Vectors, {}d".format(DIMENSION))

embeddings_dict = {}
with open("glove.6B.{}d.txt".format(DIMENSION), 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

def find_closest_vector(embedding, vocab_dict):
    return sorted(
            vocab_dict.keys(),
            key=lambda word: spatial.distance.euclidean(vocab_dict[word], embedding)
        )

def remove_furthest_words(embedding, perc_to_keep=0.2):
    # Find closest set of words
    sorted_words = find_closest_vector(embedding, embeddings_dict)
    # Select important subsection
    sorted_words = sorted_words[:int(perc_to_keep * len(sorted_words))]
    # Reorganize back into embedding dict
    selected_dict = {}
    for word in sorted_words:
        selected_dict[word] = embeddings_dict[word]
    return selected_dict

def test_dimension_flips(embedding, dict_shrink=0.002):
    if type(embedding) == str:
        embedding = embeddings_dict[embedding]
    # Vocab represents a subsection of nearer embeddings
    # as to not waste time sorting through vectors
    vocab_dict = remove_furthest_words(embedding, perc_to_keep=dict_shrink)

    # flips each dimension, if resulting vector represents a different
    # enough word, print
    def conditional_print(i, orig, embed):
        if orig != embed:
            print(">%d: %s, %s" % (i, orig, embed))
        else:
            print('>%d: ---' % i)

    original = np.copy(embedding)
    for i in range(len(embedding)):
        # Flip the desired dimension
        embedding[i] = -embedding[i]
        # Find nearest words to two vectors
        orig_as_word = find_closest_vector(original, vocab_dict)[0]
        embed_as_word = find_closest_vector(embedding, vocab_dict)[0]
        # Print only interesting information
        conditional_print(i, orig_as_word, embed_as_word)
        # Flip desired dimension back
        embedding[i] = -embedding[i]
