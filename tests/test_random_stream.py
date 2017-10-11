# -*- coding: utf-8 -*-
"""

Created on October 11, 2017

@author:  neerbek
"""
import unittest

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import tests.RunTimer

class RandomLayer:
    def __init__(self, prob, sizeX, sizeY, seed):
        self.rng = RandomStreams(seed=seed)
        self.shape = (sizeX, sizeY)
        self.prob = prob
        self.fFunction = theano.function(
            inputs=[],
            outputs=self.f()
        )

    def f(self):
        rngDropout = self.rng.binomial(n=1,
                                       size=self.shape,
                                       p=self.prob)
        rngDropout = T.cast(rngDropout, dtype='float32')
        return rngDropout

def arrayEqual(a, b, atol=0.000001):
    if a.shape[0] != b.shape[0]:
        print("a.shape[0] is different from b.shape[0]", a.shape[0], b.shape[0])
        return False
    if a.shape[1] != b.shape[1]:
        print("a.shape[1] is different from b.shape[1]", a.shape[1], b.shape[1])
        return False
    # this is slow, but ok for now
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not numpy.isclose(a[i, j], b[i, j], atol=atol):
                print("arrays differ a={:.4f}, b={:.4f}, (x,y)=({},{})".format(a[i, j], b[i, j], i, j))
                return False
    return True

class RandomStreamTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_random1(self):
        rng = RandomLayer(0.5, 4, 4, 42)
        a = rng.fFunction()
        b = rng.fFunction()
        self.assertTrue(not arrayEqual(a, b))
        pass

    def test_random2(self):
        rng1 = RandomLayer(0.5, 4, 4, 42)
        rng2 = RandomLayer(0.5, 4, 4, 42)
        a = rng1.fFunction()
        b = rng2.fFunction()
        self.assertTrue(arrayEqual(a, b))
        pass

    def test_random3(self):
        rng1 = RandomLayer(0.5, 4, 4, 42)
        rng2 = RandomLayer(0.5, 4, 4, 38346)
        a = rng1.fFunction()
        b = rng2.fFunction()
        self.assertTrue(not arrayEqual(a, b))
        pass
