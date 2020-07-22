import io
import numpy as np
import numpy.linalg as la


# Slightly modified function from the FastText documentation to add storage as vectors in numpy:
# https://fasttext.cc/docs/en/english-vectors.html
def load_vectors(fname):
    """
    Load the pretrained fasttext vectors (.vec files).
    :param fname: A string containing the file path to the model to load.
    :return: a dictionary with each word as the key and its corresponding float vector as its value.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(map(float, tokens[1:]))
    return data


def get_unit_direction(p1, p2):
    """
    Returns the normalized (length 1) unit vector of the direction between p1 and p2.
    :param p1: the first word embedding represented as a vector in numpy
    :param p2: the second word embedding represented as a vector in numpy;
                Both vectors must have the same dimension.
    :return: a unit vector in the direction of the difference between p1 and p2.
    """
    assert(p1.shape == p2.shape)
    return (p2 - p1) / la.norm(p2 - p1)


def get_projection(point, direction):
    """
    Given a unit vector and a point, return the projection of that point onto the given direction.
    :param point: a word embedding represented as a vector in numpy
    :param direction: a unit directional vector of dimension d
    :return: the projection of the word onto the direction, represented as a numpy vector
    """
    return np.dot(point, direction) * direction


def get_point_distance(points, target):
    """
    Get the distance between each point in an array and a target point.
    :param points: a numpy array of length n or greater, containing points represented as vectors of dimension d
    :param target: a point of dimension d to compare against other points, as a numpy array.
    :return: a numpy array containing n scalars, representing the distance between each point and the target
    """
    if len(points.shape) == 1:
        return la.norm(points - target)
    return la.norm(points - target, axis=1)


def get_midpoint(p1, p2):
    """
    Given two points, finds the midpoint between those points in dimension d space.
    :param p1: the first word embedding represented as a vector in numpy
    :param p2: the second word embedding represented as a vector in numpy;
                Both vectors must have the same dimension d.
    :return: a point as a numpy array of dimension d.
    """
    return p1 + ((p2 - p1) / 2)


def get_projection_matrix(points, direction):
    """
    Get the projection of each word embedding onto the given direction.
    :param points: a numpy array of length n or greater, containing points represented as vectors of dimension d
    :param direction: a unit directional vector of dimension d
    :return: a 2d numpy array of length n, containing projections for each word embedding.
    """
    # A dot d gives a vector of scalars
    # Dot product with the transpose of the direction to get the projections
    # The direction should be a unit vector, so there is no need to normalize.
    return (points.dot(direction)).dot(direction.T)


def get_orth_distance(point, direction):
    """
    Get the orthogonal distance from a point to the subspace represented by the direction vector.
    :param point: a word embedding represented as a vector of dimension d in numpy
    :param direction: a unit directional vector of dimension d
    :return: a scalar representing the distance from a word to the original direction vector
    """
    return la.norm(point - get_projection(point, direction))


def get_orth_distance_matrix(points, direction):
    """
    Get the orthogonal distance from collection of points to the subspace represented by the direction vector.
    :param points: a numpy array of length n or greater, containing points represented as vectors of dimension d
    :param direction: a unit directional vector of dimension d
    :return: a numpy array of length n, containing the distance from each point to the original direction
    """
    proj = (get_projection_matrix(points, direction))
    return la.norm(points - proj, axis=1)


def search_by_distance(points, target, method="orth", filter="distance", threshold=10.0):
    """
    A utility function for calculating various distances between a collection of words and some target vector.
    The target vector may be a point or a unit direction vector.
    :param points: a numpy array of length n or greater, containing points represented as vectors of dimension d
    :param target: a vector of dimension d, represented as a numpy array
    :param method: a string representing the kind of distance to calculate over the points.
                    If "orth", calculates the orthogonal distance between points and a direction vector.
                    If "point", calculates the Euclidean distance between points and a target point.
    :param filter: a string representing how to filter the results once distances are calculated.
                    If None, returns results for every point.
                    If "distance", returns all results below a specified Euclidean distance threshold
                    If "quantile", returns all results below a specified distance quantile;
                        (in this case, the threshold should be between 0 and 1)
                    If "limit", returns the smallest k results
                        (in this case, the threshold should be a positive integer)
    :param threshold: a value on which to filter the results of the distance calculation.
                    If the filter is None, this parameter is unused.
                    If "distance", this should be a positive continuous value representing Euclidean distance
                    If "quantile", this should be a value between 0 and 1, where 0.5 is the 50% quantile (the median)
                    If "limit", this should be a positive integer representing the smallest k values to return.
    :return: distances: a numpy array containing the scalar distances from each point to the target
             indices: returns the indices of the filtered words from the original array.
                    If filter is None, no indices are returned.
    """
    if method is "orth":
        distances = get_orth_distance_matrix(points, target)
    elif method is "point":
        distances = get_point_distance(points, target)
    elif method is "proj":
        distances = None
        # TODO: want to find each point's projected distance from the midpoint
    else:
        print(f"Illegal distance type provided: {method}")
        return None, None

    if filter is None:
        return distances, None
    elif filter is "distance":
        indices = distances <= threshold
        return distances[indices], indices
    elif filter is "quantile":
        threshold = np.quantile(distances, threshold)
        indices = distances <= threshold
        return distances[indices], indices
    elif filter is "limit":
        quantile = threshold / len(points)
        threshold = np.quantile(distances, quantile)
        indices = distances <= threshold
        return distances[indices], indices
    else:
        print(f"Illegal search method provided: {filter}")





