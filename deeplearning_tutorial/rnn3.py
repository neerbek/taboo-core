# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:03:20 2016

@author: neerbek
"""

import numpy

import theano
import theano.tensor as T

class Regression(object):
    """ Inspired by logistic_sgd.py: http://deeplearning.net/tutorial/code/logistic_sgd.py
    """
    def __init__(self, X, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.X = X

    def cost(self, y):
        return T.mean(0.5 *((self.p_y_given_x - y) ** 2))
        
    def errors(self, y):
        """fraction of errors in minibatch
        """
        y_simple = T.argmax(y, axis=1)
        # check if y has same dimension of y_pred
        if y_simple.ndim != self.y_pred.ndim:
            raise TypeError(
                'y_simple should have the same shape as self.y_pred',
                ('y_simple', y_simple.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y_simple.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y_simple))
        else:
            raise NotImplementedError()

    
class ReluLayer(object):
    def __init__(self, rng, X, n_in, n_out):
        self.X = X
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(X, self.W) + self.b
        self.output = T.nnet.relu(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]


class RNN(object):
    def __init__(self, rng, X, n_in, n_hidden, n_out):
        self.reluLayer = ReluLayer(
            rng=rng,
            X=X,
            n_in=n_in,
            n_out=n_hidden
        )
        self.regressionLayer = Regression(
            X=self.reluLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.reluLayer.W).sum()
            + abs(self.regressionLayer.W).sum()
        )
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.reluLayer.W ** 2).sum()
            + (self.regressionLayer.W ** 2).sum()
        )

        # cost of RNN is given by the cost of the output of the model, computed in the
        # regression layer
        self.cost = (self.regressionLayer.cost)
        # same holds for the function computing the number of errors
        self.errors = self.regressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.reluLayer.params + self.regressionLayer.params
        self.X = X
        
    def get_representation(self, left_in, right_in):
        return self.reluLayer.output.eval({ self.X : numpy.concatenate([left_in, right_in]).reshape(1, 2*left_in.shape[1])})