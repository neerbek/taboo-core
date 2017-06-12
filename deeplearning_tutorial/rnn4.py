# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:03:20 2016

@author: neerbek
"""

import numpy

import theano
#import theano.printing as printing
#import theano.function as function
import theano.tensor as T

class Regression(object):
    """ Inspired by logistic_sgd.py: http://deeplearning.net/tutorial/code/logistic_sgd.py
    """
    def __init__(self, rng, X, n_in, n_out, cost_weight=0):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        #self.W = theano.shared(value=W_values, name='W_reg', borrow=True)
        self.W = theano.shared(value=W_values, name='W_reg', borrow=False)
        # initialize the biases b as a vector of n_out 0s
        b_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_out)),
                high=numpy.sqrt(6. / (n_out)),
                size=(n_out,)
            ),
            dtype=theano.config.floatX
        )
        #self.b = theano.shared(value=b_values, name='b_reg', borrow=True)
        self.b = theano.shared(value=b_values, name='b_reg', borrow=False)
        self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
        self.cost_weight = cost_weight
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.X = X

        
    def cost(self, y):
        err = (self.p_y_given_x - y)
        cost_weight = T.ones_like(y.shape) + y * self.cost_weight
        err_weighted = err * cost_weight
        return T.mean(0.5 *((err_weighted) ** 2))

    def softmax_debug(self, w):
        maxes = numpy.amax(w, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = numpy.exp(w - maxes)
        dist = e / numpy.sum(e, axis=1, keepdims=True)
        return dist

    def cost_debug(self, X, y):
        p_y_given_x = self.softmax_debug(numpy.dot(X, self.W.get_value()) + self.b.get_value())
        err = p_y_given_x - y
        cost_weight = numpy.ones(shape=y.shape) + y * self.cost_weight
        #print("err shape", err.shape)
        err_weighted = numpy.multiply(err, cost_weight)
        #print("err_weighted shape1", err_weighted.shape)
        return numpy.mean(0.5*((err_weighted) **2))
        
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

#for dropout:
#https://blog.wtf.sg/2014/07/23/dropout-using-theano/
#http://stackoverflow.com/questions/29540592/why-does-my-dropout-function-in-theano-slow-down-convolution-greatly    
class ReluLayer(object):
    def __init__(self, rng, X, Z, n_in, n_out):
        self.X = X
        self.Z = Z
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
#        self.W = theano.shared(value=W_values, name='W_relu', borrow=True)
        self.W = theano.shared(value=W_values, name='W_relu', borrow=False)
        b_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_out)),
                high=numpy.sqrt(6. / (n_out)),
                size=(n_out,)
            ),
            dtype=theano.config.floatX
        )
#        self.b = theano.shared(value=b_values, name='b_relu', borrow=True)
        self.b = theano.shared(value=b_values, name='b_relu', borrow=False)
        lin_output = T.dot(X, self.W) + self.b
        self.output = T.nnet.relu(lin_output)
        #dropout
        self.output = self.output*Z  #element-wise mult
        # parameters of the model
        self.params = [self.W, self.b]


class RNN(object):
    def __init__(self, rng, X, Z, n_in, n_hidden, n_out, cost_weight=0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.reluLayer = ReluLayer(
            rng=rng,
            X=X,
            Z=Z,
            n_in=n_in,
            n_out=n_hidden
        )
        self.regressionLayer = Regression(
            rng=rng,
            X=self.reluLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            cost_weight=cost_weight
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

        #A NxM matrix of probabilities
        self.p_y_given_x = self.regressionLayer.p_y_given_x
        #A list of N indexes (e.g. predictions of values of y)
        self.y_pred = self.regressionLayer.y_pred
        # cost of RNN is given by the cost of the output of the model, computed in the
        # regression layer
        self.cost = (self.regressionLayer.cost)
        # same holds for the function computing the number of errors
        self.errors = self.regressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        self.params.extend(self.reluLayer.params)
        self.params.extend(self.regressionLayer.params)
        
        self.X = X
        
    def get_representation(self, left_in, right_in):
        return self.reluLayer.output.eval({ self.X : numpy.concatenate([left_in, right_in]).reshape(1, 2*left_in.shape[1])})
        
        
from six.moves import cPickle
VERSION_1="RNN_SERIALIZED_VERSION_1"
VERSION="RNN_SERIALIZED_VERSION_2"
def get_object_list(reg, epoch, acc):
    obj_list = [ VERSION, "{}".format(epoch), "{:.4f}".format(acc), reg.reluLayer.W, reg.reluLayer.b, reg.regressionLayer.W, reg.regressionLayer.b]
    return obj_list

def save(rnn, filename='model.save', epoch=0, acc=0):
    obj_list = get_object_list(rnn, epoch, acc)
    f = open(filename, 'wb')
    for obj in obj_list[:3]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    for obj in obj_list[3:]:  #theano tensor variables
        cPickle.dump(obj.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load(rnn, filename='model.save'):
    epoch = 0
    acc = 0
    obj_list = get_object_list(rnn, epoch, acc)
    f = open(filename, 'rb')
    for i in range(len(obj_list)):
        v = cPickle.load(f)
        obj_list[i] = v
    if obj_list[0]!=VERSION:
        raise Exception("Version mismatch in rnn4.load")
    epoch = int(obj_list[1])
    acc =  float(obj_list[2])
    #need to cast with astype if it was saved with different bit width (32 vs 64)
    rnn.reluLayer.W.set_value(numpy.array(obj_list[3]).astype(dtype=theano.config.floatX))
    rnn.reluLayer.b.set_value(numpy.array(obj_list[4]).astype(dtype=theano.config.floatX))
    rnn.regressionLayer.W.set_value(numpy.array(obj_list[5]).astype(dtype=theano.config.floatX))
    rnn.regressionLayer.b.set_value(numpy.array(obj_list[6]).astype(dtype=theano.config.floatX))
    #print("W[5,10] ", reg.reluLayer.W.get_value()[5,10])
    f.close()
    return (epoch, acc)

def load_v1(rnn, filename='model.save'):
    epoch = 0
    acc = 0
    obj_list = get_object_list(rnn, epoch, acc)
    f = open(filename, 'rb')
    for i in range(len(obj_list)):
        v = cPickle.load(f)
        obj_list[i] = v
    if obj_list[0]!=VERSION_1:
        raise Exception("Version mismatch in rnn4.load")
    epoch = obj_list[1]
    acc =  obj_list[2]
    rnn.reluLayer.W.set_value(obj_list[3])
    rnn.reluLayer.b.set_value(obj_list[4])
    rnn.regressionLayer.W.set_value(obj_list[5])
    rnn.regressionLayer.b.set_value(obj_list[6])
    #print("W[5,10] ", reg.reluLayer.W.get_value()[5,10])
    f.close()
    return (epoch, acc)

def load_v0(reg, name='model.save'):
    epoch = 0
    acc = 0
    obj_list = get_object_list(reg, epoch, acc)
    obj_list = obj_list[3:]
    f = open(name, 'rb')
    for i in range(len(obj_list)):
        v = cPickle.load(f)
        obj_list[i] = v
    reg.reluLayer.W.set_value(obj_list[0])
    reg.reluLayer.b.set_value(obj_list[1])
    reg.regressionLayer.W.set_value(obj_list[2])
    reg.regressionLayer.b.set_value(obj_list[3])
    #print("W[5,10] ", reg.reluLayer.W.get_value()[5,10])
    f.close()
    return (epoch, acc)

