# -*- coding: utf-8 -*-
"""
Created on Thu 27 2017

@author: neerbek
"""

import unittest

import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T


import tests.RunTimer
import embedding

def mult_embedding(e1, e2):
    if len(e1) != len(e2):
        raise Exception("e1 and e2 need to have same lengths {}, {}".format(len(e1), len(e2)))
    res = 0
    for i in range(len(e1)):
        res += e1[i] * e2[i]
    return res


class EmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_serialize(self):
        emb = [42]
        s = embedding.serialize_embedding(emb)
        self.assertEqual("[42]", s)

    def test_deserialize(self):
        s = "[42.124]"
        emb = embedding.deserialize_embedding(s)
        self.assertEqual(len(emb), 1)
        self.assertEqual(emb[0], 42.124)

    def test_mult(self):
        e1 = [2, 4]
        e2 = [1, 3]
        s = mult_embedding(e1, e2)
        self.assertEqual(14, s)

    def test_mult_matrix(self):
        e1 = [2, 4]
        e2 = [1, 3]
        a1 = numpy.array([e1, e2])
        e11 = [1.5, 2]
        e12 = [5, -1]
        a2 = numpy.array([e11, e12])
        res = embedding.mult_embedding_numpy_matrix(a1, a2)
        expected_res = numpy.array([[mult_embedding(e1, e11), mult_embedding(e1, e12)], [mult_embedding(e2, e11), mult_embedding(e2, e12)]])
        self.assertTrue(numpy.array_equiv(res, expected_res), "Calculated dot products check failed. {} {}".format(res, expected_res))

    def cos_distance_matrix1(self, get_distances):
        e1 = [-3, 0]
        e2 = [0, 3]
        a1 = numpy.array([e1, e2])
        d = get_distances(a1)
        self.assertTrue(d.shape[0] == a1.shape[0])
        self.assertTrue(d.shape[1] == a1.shape[0])
        self.assertEqual(0, d[0, 0])
        self.assertEqual(0, d[1, 1])
        self.assertTrue(d[0, 1] == d[0, 1])
        self.assertTrue(d[0, 1] == 0)

    def test_cos_distance_numpy_matrix1(self):
        self.cos_distance_matrix1(embedding.cos_distance_numpy_matrix)

    def test_cos_distance_theano_matrix1(self):
        x = T.dmatrix('x')
        self.cos_distance_matrix1(theano.function([x], embedding.cos_distance_theano_matrix(x)))

    def cos_distance_matrix2(self, get_distances):
        e1 = [3, 0]
        e2 = [3, 3]
        a1 = numpy.array([e1, e2])
        d = get_distances(a1)
        self.assertTrue(d.shape[0] == a1.shape[0])
        self.assertTrue(d.shape[1] == a1.shape[0])
        self.assertAlmostEqual(d[0, 0], 0, places=8)
        self.assertAlmostEqual(d[1, 1], 0, places=8)
        self.assertAlmostEqual(d[0, 1], d[0, 1], places=8)
        self.assertAlmostEqual(d[0, 1], 1 / numpy.sqrt(2), places=8)

    def test_cos_distance_numpy_matrix2(self):
        self.cos_distance_matrix2(embedding.cos_distance_numpy_matrix)

    def test_cos_distance_theano_matrix2(self):
        x = T.dmatrix('x')
        self.cos_distance_matrix2(theano.function([x], embedding.cos_distance_theano_matrix(x)))

    def cos_distance_matrix3(self, get_distances):
        e1 = [-3, 3]
        e2 = [3, 3]
        a1 = numpy.array([e1, e2])
        d = get_distances(a1)
        self.assertTrue(d.shape[0] == a1.shape[0])
        self.assertTrue(d.shape[1] == a1.shape[0])
        self.assertAlmostEqual(d[0, 0], 0, places=8)
        self.assertAlmostEqual(d[1, 1], 0, places=8)
        self.assertAlmostEqual(d[0, 1], d[0, 1], places=8)
        self.assertAlmostEqual(d[0, 1], 0, places=8)

    def test_cos_distance_numpy_matrix3(self):
        self.cos_distance_matrix3(embedding.cos_distance_numpy_matrix)

    def test_cos_distance_theano_matrix3(self):
        x = T.dmatrix('x')
        self.cos_distance_matrix3(theano.function([x], embedding.cos_distance_theano_matrix(x)))

    def test_get_min(self):
        e1 = [3, 4]
        e2 = [4, 2]
        a1 = numpy.array([e1, e2])
        d = embedding.get_closest_embedding(a1)
        self.assertTrue(type(d) is tuple, "unexpected type {}".format(type(d)))
        self.assertEqual(2, len(d))
        self.assertEqual((0, 1), d, "Tuple is wrong {}".format(d))

    def test_get_min2(self):
        e1 = [3, 4]
        e2 = [4, 2]
        a1 = numpy.array([e1, e2])
        d = embedding.get_closest_embedding(a1, 1)
        self.assertTrue(type(d) is tuple, "unexpected type {}".format(type(d)))
        self.assertEqual(2, len(d))
        self.assertEqual((1, 0), d, "Tuple is wrong {}".format(d))

    def test_get_min3(self):
        e1 = [3, 1]
        e2 = [1, 2]
        a1 = numpy.array([e1, e2])
        x = T.dmatrix('x')
        get_distances = theano.function([x], embedding.cos_distance_theano_matrix(x))
        d = get_distances(a1)
        pair = embedding.get_closest_embedding(d, 1)
        self.assertTrue(type(pair) is tuple, "unexpected type {}".format(type(pair)))
        self.assertEqual(2, len(pair))
        self.assertEqual((1, 0), pair, "Tuple is wrong {}".format(pair))
        self.assertAlmostEqual(0.70710678, d[1, 0], places=7)

    def test_get_min4(self):
        e1 = [3, 1]
        e2 = [1, 2]
        a1 = numpy.array([e1, e2])
        x = T.dmatrix('x')
        get_distances = theano.function([x], embedding.cos_distance_theano_matrix(x))
        d = get_distances(a1)
        d = embedding.get_closest_embedding(d)
        self.assertTrue(type(d) is tuple, "unexpected type {}".format(type(d)))
        self.assertEqual(2, len(d))
        self.assertEqual((0, 1), d, "Tuple is wrong {}".format(d))

    def are_columns_sorted(self, m):
        (numRows, numColumns) = m.shape
        for j in range(numColumns):
            prev = 1
            for i in range(numRows):
                current = m[i, j]
                if current > prev:
                    return False
                prev = current
        return True

    def test_indices(self):
        rng = RandomState(1234)
        weights = rng.uniform(0, 1, size=6 * 6)
        weights = weights.reshape(6, 6)
        self.assertTrue(self.are_columns_sorted(weights) is False)
        t2 = embedding.get_sort_indices_desc(weights)
        self.assertEqual((6, 6), t2.shape)
        t3 = embedding.apply_sort_indices(weights, t2)
        self.assertEqual((6, 6), t3.shape)
        self.assertTrue(self.are_columns_sorted(t3))

    def test_indices2(self):
        rng = RandomState(473526)
        l = 200
        weights = rng.uniform(0, 1, size=l * l)
        weights = weights.reshape(l, l)
        self.assertTrue(self.are_columns_sorted(weights) is False)
        t2 = embedding.get_sort_indices_desc(weights)
        self.assertEqual((l, l), t2.shape)
        t3 = embedding.apply_sort_indices(weights, t2)
        self.assertEqual((l, l), t3.shape)
        self.assertTrue(self.are_columns_sorted(t3))


if __name__ == "__main__":
    unittest.main()
