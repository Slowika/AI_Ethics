from utilities import *

DEFAULT_FILTER_PARAMS = {
    "method": "quantile",
    "threshold": 0.01
}


def run_projection_experiment(words, embeddings, word1, word2, filter_params=DEFAULT_FILTER_PARAMS):
    """
    A function for calculating distances and projections between two words in the embedded space.
    :param words: An array of strings (corresponding to the keys of the extracted word embeddings) of length n
    :param embeddings: A numpy array of length n containing the vectors of the corresponding words
    :param word1: a word embedding represented as a numpy array
    :param word2: a word embedding represented as a numpy array
    :param filter_params: a dictionary of parameters on which to filter the results of the distance calculation
    :return: direction: a unit vector in the direction of the comparison
             distances: a numpy array containing the scalar distances from each point to the target
             filtered_words: an array of words which has been filtered, corresponding to the distances
             projections: an array of length n, each row containin a vector of the projection onto the direction

    """
    direction = get_unit_direction(word1, word2)

    projections = get_projection_matrix(embeddings, direction)
    distances, indices = search_by_distance(embeddings, direction, method=filter_params["method"],
                                            threshold=filter_params["threshold"])
    if indices is not None:
        filtered_words = words[indices]
    else:
        filtered_words = words

    return direction, distances, filtered_words, projections




