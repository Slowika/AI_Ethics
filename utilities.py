import io
import numpy as np
import numpy.linalg as la


# Function from the FastText documentation:
# https://fasttext.cc/docs/en/english-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def get_direction(p1, p2):
    return (p2 - p1) / la.norm(p2 - p1)


def get_projection(point, direction):
    return np.dot(point, direction) * direction


def get_point_distance(points, target):
    if len(points.shape) == 1:
        return la.norm(points - target)
    return la.norm(points - target, axis=1)


def get_midpoint(p1, p2):
    return p1 + ((p2 - p1) / 2)


def get_projection_matrix(points, direction):
    # A dot d gives a vector of scalars
    # Dot product with the transpose of the direction to get the projections
    return (points.dot(direction)).dot(direction.T)


def get_orth_distance(point, direction):
    return la.norm(point - get_projection(point, direction))


def get_orth_distance_matrix(points, direction):
    proj = (get_projection_matrix(points, direction))
    return la.norm(points - proj, axis=1)


def search_by_distance(points, target, method="orth", filter="distance", threshold=10.0):
    if method is "orth":
        distances = get_orth_distance_matrix(points, target)
    elif method is "point":
        distances = get_point_distance(points, target)
    elif method is "proj":
        distances = None
        # TODO
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





