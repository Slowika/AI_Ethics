# import fasttext
from gensim.models import KeyedVectors
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

en_model = KeyedVectors.load_word2vec_format('wiki.en/wiki-news-300d-1M.vec')
sw_model = KeyedVectors.load_word2vec_format('wiki.en/wiki-news-300d-1M-subword.vec')


def word_similarity(word):
    similar_word = en_model.most_similar(positive=[word])
    print("Non-subword result:")
    for i in range(5):
        print("{} ({:.2%})".format(
            similar_word[i][0], similar_word[i][1]))

    similar_word = sw_model.most_similar(positive=[word])
    print("\nSubword result:")
    for i in range(5):
        print("{} ({:.2%})".format(
            similar_word[i][0], similar_word[i][1]))


# prints the similarity percentages for two words given a comparison word
def compare(worda, wordb, com_word):
    print("Non-subword result:")
    print("{} and {}: {:.2%}".format(worda, com_word, en_model.similarity(worda, com_word)))
    print("{} and {}: {:.2%}".format(wordb, com_word, en_model.similarity(wordb, com_word)))

    print("\nSubword result:")
    print("{} and {}: {:.2%}".format(worda, com_word, sw_model.similarity(worda, com_word)))
    print("{} and {}: {:.2%}".format(wordb, com_word, sw_model.similarity(wordb, com_word)))


# A is to B as C is to D
def word_analogy(worda, wordb, wordc):
    print("{} is to {} as {} is to {}".format(worda, wordb, wordc,
                                              en_model.most_similar(negative=[worda], positive=[wordb, wordc])[0][0]))


def sw_word_analogy(worda, wordb, wordc):
    print("{} is to {} as {} is to {}".format(worda, wordb, wordc,
                                              sw_model.most_similar(negative=[worda], positive=[wordb, wordc])[0][0]))
