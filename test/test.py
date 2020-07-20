from __future__ import print_function
# import fasttext
from gensim.models import KeyedVectors, word2vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

# plots the closest words to a given word
def plot_TSNE(word, limit=200, model=en_model):
    words = []
    embedding = np.array([])
    filename = 'results/tsne_{}_reg.png'.format(word)
    vector_dim = 300

    if model == 'sw':
        model = sw_model
        filename = 'results/tsne_{}_sw.png'.format(word)

    list_words = model.most_similar(positive=[word], topn=limit)
    words.append(word)
    embedding = np.append(embedding, model[word])

    for word in list_words:
        words.append(word[0])

        embedding = np.append(embedding, model[word[0]])

    embedding = embedding.reshape(limit + 1, vector_dim)

    tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

    low_dim_embedding = tsne.fit_transform(embedding)

    assert low_dim_embedding.shape[0] >= len(words), "More labels than embeddings"
    plt.figure(figsize=(22, 22), dpi=300)  # in inches
    for i, label in enumerate(words):
        x, y = low_dim_embedding[i, :]
        plt.scatter(x, y, c='red')
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
