# -*- coding: utf-8 -*-
"""
Created Jun 21 13:14 2017

@author: neerbek
"""
import numpy
import theano
import theano.tensor as T


# https://stackoverflow.com/questions/29365370/how-to-code-adagrad-in-python-theano
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py#L385
def adagrad(params, grads, lr, epsilon=1e-8):
    updates = {}
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(
            numpy.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable,
            name=param.name + "_accu")
        accu_new = accu + grad**2  # this is a function update to be used several times
        updates[accu] = accu_new  # one per param
        updates[param] = param - (lr * grad / T.sqrt(accu_new + epsilon))
    return updates

class AdagradLearner:
    def __init__(self, lr, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon

    def getUpdates(self, params, cost):
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        return adagrad(params, grads, self.lr, self.epsilon)


def gd(params, grads, lr):  # gradient decent
    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - (lr * grad)
    return updates

class GradientDecentLearner:
    def __init__(self, lr):
        self.lr = lr

    def getUpdates(self, params, cost):
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        return gd(params, grads, self.lr)

def gd_momentum(params, grads, lr, mc):  # gradient decent with momentum
    updates = {}
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        param_prev = theano.shared(
            numpy.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable,
            name=param.name + "_prev")
        updates[param] = param - (lr * grad) - mc * param_prev
        # or updates[param] = param - (lr * (1 - mc) *  grad) - mc*param_prev
        # https://www.mathworks.com/help/nnet/ref/traingdm.html
        updates[param_prev] = param
    return updates

class GradientDecentWithMomentumLearner:
    def __init__(self, lr, mc):
        self.lr = lr
        self.mc = mc

    def getUpdates(self, params, cost):
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        return gd_momentum(params, grads, self.lr, self.mc)
