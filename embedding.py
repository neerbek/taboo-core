#  -*- coding: utf-8 -*-
"""
Created on Thu Jul 27

@author: neerbek
"""
import numpy
import theano.tensor as T

def serialize_embedding(emb):
    "returns a string representation of the embedding"
    if emb is None:
        return None
    arr = "["
    for e in emb:
        arr += str(e) + ","
    arr = arr[:-1] + "]"
    return arr

def deserialize_embedding(s):
    "deserialize a string to an array of numbers"
    s = s[1:-1]  # remove "[" + "]"
    entries = s.split(",")
    res = []
    for emb in entries:
        if len(emb) == 0:
            print("warning empty component found in serialized embedding")
            continue
        f = float(emb)
        res.append(f)
    return res

def mult_embedding_numpy_matrix(m1, m2):
    "Assumes that the rows are embeddings for both matrices"
    # the two checks below does not check all rows/columns
    if m1.shape != m2.shape:
        raise Exception("a1 and a2 need to have same shape {}, {}".format(m1.shape, m2.shape))
    res = numpy.dot(m1, numpy.transpose(m2))
    return res

def cos_distance_theano_matrix(m1):
    """Theano impl of cos_distance_numpy_matrix
       See cos_distance_numpy_matrix"""
    d = T.sum(m1 * m1, axis=1)               # vector of ||v_i||^2
    d = T.sqrt(d)                            # vector of ||v_i||
    res = m1 / d[:, None]                    # rows are divided by ||v_i||
    res = T.dot(res, T.transpose(res))   # normalized v_i, v_j, component (i,j) = v_i \dot v_j
    res = T.extra_ops.fill_diagonal(res, 0)  # note different than for numpy
    return res

def cos_distance_numpy_vector(v1, v2):
    """get cos angle (similarity) between two vectors"""
    d1 = numpy.sum(v1 * v1)
    d1 = numpy.sqrt(d1)      # magnitude of v1
    d2 = numpy.sum(v2 * v2)
    d2 = numpy.sqrt(d2)      # magnitude of v2
    n1 = v1 / d1
    n2 = v2 / d2
    return numpy.sum(n1 * n2)


def cos_distance_numpy_matrix(m1):
    """Assumes that the rows are embeddings.
       Returns a matrix of cosine angle for each pair of embeddings"""
    d1 = numpy.sum(m1 * m1, axis=1)   # vector of ||v_i||^2
    d1 = numpy.sqrt(d1)                # vector of ||v_i||
    # res = (m1.T / d).T             # rows are divided by ||v_i||
    n1 = m1 / d1[:, None]            # rows are divided by ||v_i||, faster by ~2% for medium sized matrix
    # see https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    res = numpy.dot(n1, numpy.transpose(n1))  # normalized v_i, v_j, component (i,j) = v_i \dot v_j
    numpy.fill_diagonal(res, 0)                 # we are not interested in diagonal components
    return res

# old impl gives inf for large vectors
def cos_distance_numpy_matrix2(m1):
    """Assumes that the rows are embeddings.
       Returns a matrix of cosine angle for each pair of embeddings"""
    res = numpy.dot(m1, numpy.transpose(m1))
    d = numpy.diagonal(res)  # vector of ||v_i||^2
    d = numpy.sqrt(d)        # vector of ||v_i||
    d = numpy.outer(d, d)
    numpy.fill_diagonal(res, 0)
    return res / d

def cos_distance_numpy_two_matrices(m1, m2):
    """Assumes that the rows are embeddings.
       Returns a matrix of cosine angle for each pair of embeddings"""
    d1 = numpy.sum(m1 * m1, axis=1)   # vector of ||v_i||^2
    d1 = numpy.sqrt(d1)                # vector of ||v_i||
    d2 = numpy.sum(m2 * m2, axis=1)   # vector of ||v_i||^2
    d2 = numpy.sqrt(d2)                # vector of ||v_i||
    # res = (m1.T / d).T             # rows are divided by ||v_i||
    n1 = m1 / d1[:, None]            # rows are divided by ||v_i||, faster by ~2% for medium sized matrix
    n2 = m2 / d2[:, None]            # rows are divided by ||v_i||, faster by ~2% for medium sized matrix
    # see https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    res = numpy.dot(n1, n2.T)  # normalized v_i, v_j, component (i,j) = v_i \dot v_j
    return res

def get_closest_embedding(dist, emb_index=None):
    """Returns coordinates for max index. In case there are multiple maximum values we only return
       indices for the first one.
       Diagonal entries should be 0 otherwise they will be maximum"""
    a = dist
    if emb_index is not None:
        a = dist[emb_index, :]
    pair = numpy.unravel_index(a.argmax(), a.shape)
    if emb_index is not None:
        pair = (emb_index, pair[0])
    return pair

def get_sort_indices_desc(dist):
    """Returns coordinates for max indexes. Returns a matrix of column indices such that each column is the sorted list of indices of the corresponding column in the dist matrix.
       Diagonal entries should be 0 otherwise they will be maximum"""
    return numpy.argsort(-dist, axis=0)  # sort every column return indices of sort, -dist means biggest first

def get_sort_indices_asc(dist):
    """Returns coordinates for max indexes. Returns a matrix of column indices such that each column is the sorted list of indices of the corresponding column in the dist matrix.
       Diagonal entries should be 0 otherwise they will be maximum"""
    return numpy.argsort(dist, axis=0)  # sort every column return indices of sort

def apply_sort_indices(dist, indices):
    """Returns a sorted copy of dist, where each column is ordered according to indices"""
    l = dist.shape[1]
    res = []
    for i in range(l):
        tmp = dist[:, i][indices[:, i]]  # I would like to do this on whole matrix dist at once
        res.append(tmp)                  # but my numpy-fu is only strong enough for column based
    t3 = numpy.array(res)                # indexing. Which means array have to do a concatanation here.
    t3 = t3.T                            # array concatanates the "wrong" way
    return t3

def euclid_distance_numpy_vector(v1, v2):
    """get euclidean distance between"""
    d1 = v1 - v2
    d1 = d1 * d1  # squared
    s = numpy.sum(d1)  # summed
    return numpy.sqrt(s)

