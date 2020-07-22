from utilities import *
import matplotlib.pyplot as plt
#import fasttext as ft

DEFAULT_FILTER_PARAMS = {
    "method": "quantile",
    "threshold": 0.01
}


def run_projection_experiment(words, embeddings, word1, word2, filter_params=DEFAULT_FILTER_PARAMS):
    direction = get_direction(word1, word2)

    projections = get_projection_matrix(embeddings, direction)
    distances, indices = search_by_distance(embeddings, direction, method=filter_params["method"],
                                            threshold=filter_params["threshold"])
    if indices is not None:
        filtered_words = words[indices]
    else:
        filtered_words = words

    return distances, filtered_words, projections




