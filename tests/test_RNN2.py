# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:15:04 2017

@author: neerbek
"""
# import os
# os.chdir("/home/neerbek/jan/phd/DLP/paraphrase/taboo-core")

# (local-set-key (kbd "C-,") (quote elpy-flymake-next-error))

import unittest
import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T

import rnn_model.rnn as nn_model

import tests.RunTimer


class RNNTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def weighted_cost_eval(self, get_cost, values):
        rng = RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)

        rng = RandomState(1234)
        x_val = rng.randint(10, size=(50, 10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50, 5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        c = get_cost(reg, x_val, y_val)
        self.assertAlmostEqual(values[0], c, places=4)
        reg.cost_weight = [1, 0, 0, 0, 1]
        c = get_cost(reg, x_val, y_val)
        self.assertAlmostEqual(values[1], c, places=4)
        reg.cost_weight = 2
        c = get_cost(reg, x_val, y_val)
        self.assertAlmostEqual(values[2], c, places=4)
        reg.cost_weight = 1
        c = get_cost(reg, x_val, y_val)
        self.assertAlmostEqual(values[3], c, places=4)
        reg.cost_weight = [1, 1, 1, 1, 1]
        c = get_cost(reg, x_val, y_val)
        self.assertAlmostEqual(values[3], c, places=4)

    def get_RMS_cost(reg, x_val, y_val):
        x = reg.X
        y = T.matrix('y', dtype=theano.config.floatX)
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        return c

    def get_RMS_cost_debug(reg, x_val, y_val):
        return reg.cost_RMS_debug(x_val, y_val)

    def test_cross_entropy(self):
        rng = RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)

        rng = RandomState(1234)
        x_val = rng.randint(10, size=(50, 10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(2, size=50)
        y_val = numpy.zeros(shape=(50, 5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][0] = 1 - truth_val[i]
            y_val[i][4] = truth_val[i]
        y_val = y_val.astype(dtype=theano.config.floatX)
        get_probs = theano.function(inputs=[x], outputs=reg.p_y_given_x)
        p_y = get_probs(x_val)
        cost_debug = 0
        for i in range(50):
                cost_debug += -y_val[i][0] * numpy.log(p_y[i][0])
                cost_debug += - y_val[i][4] * numpy.log(p_y[i][4])
        cost_debug /= 50   # mean over 50 rows
        cost = reg.cost_cross(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(cost_debug, c, places=12)
        c = reg.cost_cross_debug(x_val, y_val)
        self.assertAlmostEqual(cost_debug, c, places=12)
        reg.cost = reg.cost_cross
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(cost_debug, c, places=12)

        reg.cost_weight = [0.25, 0, 0, 0, 4]
        cost_debug = 0
        for i in range(50):
            cost_debug += -1.25 * y_val[i][0] * numpy.log(
                p_y[i][0]) - 5 * y_val[i][4] * numpy.log(p_y[i][4])
        cost_debug /= 50   # mean over 50 rows
        cost = reg.cost_cross(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(cost_debug, c, places=12)
        c = reg.cost_cross_debug(x_val, y_val)
        self.assertAlmostEqual(cost_debug, c, places=12)

    def test_weighted_cost6(self):
        self.weighted_cost_eval(RNNTest.get_RMS_cost,
                                [0.1664, 0.2084, 0.8376, 0.4181])

    def test_weighted_cost7(self):
        self.weighted_cost_eval(RNNTest.get_RMS_cost_debug,
                                [0.1664, 0.2084, 0.8376, 0.4181])

    def test_weighted_cost2(self):
        rng = RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)

        rng = RandomState(1234)
        x_val = rng.randint(10, size=(50, 10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50, 5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = [1, 1, 1, 1, 1]
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_RMS_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)

    def test_weighted_cost3(self):
        rng = RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)

        rng = RandomState(1234)
        x_val = rng.randint(10, size=(50, 10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50, 5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = 1
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_RMS_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)

    def test_weighted_cost4(self):
        rng = RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 4, 2)
        x_val = numpy.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                             [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0]])
        print(x_val.shape)
        truth_val = numpy.array([1, 0, 1, 0, 1, 0])
        x_val = x_val.astype(dtype=theano.config.floatX)
        y_val = numpy.zeros(
            shape=(x_val.shape[0], 2), dtype=theano.config.floatX)
        for i in range(y_val.shape[0]):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = 1
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)
        c = reg.cost_RMS_debug(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)

        reg.cost_weight = [1, 1]
        cost = reg.cost(y)
        cost_model = theano.function(inputs=[x, y], outputs=cost)
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)
        c = reg.cost_RMS_debug(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)

    def calc_cost(self, y, p, cw):
        err = p - y  # [[-0.5, 0.5], [-0.8, 0.8]]
        cost_weight = numpy.ones(shape=y.shape) + y * cw
        err_weighted = err * cost_weight  # numpy.multiply(err, cost_weight)
        c = numpy.mean(0.5 * ((err_weighted)**2))
        return c

    def test_weighted_cost5(self):
        y = numpy.array([[1, 0], [1, 0]])
        p = numpy.array([[0.5, 0.5], [0.2, 0.8]])
        cw = [0, 1]
        c = self.calc_cost(y, p, cw)
        self.assertAlmostEqual(0.2225, c, places=4)
        cw = 0
        c = self.calc_cost(y, p, cw)
        self.assertAlmostEqual(0.2225, c, places=4)
        cw = [1, 0]
        c = self.calc_cost(y, p, cw)
        self.assertAlmostEqual(
            0.25 * 0.5 * (1 + 0.25 + 1.6**2 + 0.8**2), c, places=4)


if __name__ == "__main__":
    unittest.main()
