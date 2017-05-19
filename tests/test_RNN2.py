# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:15:04 2017

@author: neerbek
"""
import unittest
import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T

import deeplearning_tutorial.rnn4 as nn_model

import tests.RunTimer

class RNNTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)


    def test_weighted_cost(self):
        rng=RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)  
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)
        
        rng=RandomState(1234)
        x_val = rng.randint(10, size=(50,10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50,5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.1664, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.1664, c, places=4)
        reg.cost_weight = [1,0,0,0,1]
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.2084, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.2084, c, places=4)
        reg.cost_weight = 2
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.8376, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.8376, c, places=4)
        reg.cost_weight = 1
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        reg.cost_weight = [1,1,1,1,1]
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)

    def test_weighted_cost2(self):
        rng=RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)  
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)
        
        rng=RandomState(1234)
        x_val = rng.randint(10, size=(50,10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50,5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = [1,1,1,1,1]
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)

    def test_weighted_cost3(self):
        rng=RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)  
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 10, 5)
        
        rng=RandomState(1234)
        x_val = rng.randint(10, size=(50,10))
        x_val = x_val.astype(dtype=theano.config.floatX)
        truth_val = rng.randint(5, size=50)
        y_val = numpy.zeros(shape=(50,5), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = 1
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.4181, c, places=4)

    def test_weighted_cost4(self):
        rng=RandomState(1234)
        x = T.matrix('x', dtype=theano.config.floatX)  
        y = T.matrix('y', dtype=theano.config.floatX)
        reg = nn_model.Regression(rng, x, 4, 2)
        x_val = numpy.array([[0,0,0,1],
                 [0,0,1,0],
                 [0,0,1,1],
                 [0,1,0,0],                 
                 [0,1,0,1],                 
                 [0,1,1,0]])
        print(x_val.shape)               
        truth_val = numpy.array([1,0,1,0,1,0])
        x_val = x_val.astype(dtype=theano.config.floatX)
        y_val = numpy.zeros(shape=(x_val.shape[0],2), dtype=theano.config.floatX)
        for i in range(y_val.shape[0]):
            y_val[i][truth_val[i]] = 1
        y_val = y_val.astype(dtype=theano.config.floatX)
        reg.cost_weight = 1
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)

        reg.cost_weight = [1,1]
        cost = reg.cost(y)
        cost_model = theano.function(
            inputs=[x,y],
            outputs=cost
        )
        c = cost_model(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)
        c = reg.cost_debug(x_val, y_val)
        self.assertAlmostEqual(0.2050, c, places=4)
            

    def calc_cost(self, y,p, cw):
        err = p - y   #[[-0.5, 0.5], [-0.8, 0.8]]
        cost_weight = numpy.ones(shape=y.shape) + y * cw
        err_weighted = numpy.multiply(err, cost_weight)
        c = numpy.mean(0.5*((err_weighted) **2))
        return c
        
    def test_weighted_cost5(self):
        y = numpy.array([[1, 0], [1, 0]])
        p = numpy.array([[0.5,0.5], [0.2, 0.8]])
        cw = [0,1]
        c = self.calc_cost(y,p,cw)
        self.assertAlmostEqual(0.2225, c, places=4)
        cw = 0
        c = self.calc_cost(y,p,cw)
        self.assertAlmostEqual(0.2225, c, places=4)
        cw = [1,0]
        c = self.calc_cost(y,p,cw)
        self.assertAlmostEqual( 0.25*0.5 * (1 + 0.25 + 1.6**2 + 0.8**2), c, places=4)
        


if __name__ == "__main__":
    unittest.main()
