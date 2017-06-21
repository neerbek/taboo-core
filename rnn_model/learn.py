# -*- coding: utf-8 -*-
"""
Created Jun 21 13:14 2017

@author: neerbek
"""
import numpy
import theano
import theano.tensor as T

#https://stackoverflow.com/questions/29365370/how-to-code-adagrad-in-python-theano
#https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py#L385
def adagrad(params, grads, lr, epsilon = 1e-8):
    updates = {}
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2  #this is a function update to be used several times
        updates[accu] = accu_new     #one per param
        updates[param] = param - (lr * grad /
                                  T.sqrt(accu_new + epsilon))
    return updates

def gd(params, grads, lr):  #gradient decent
    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - (lr * grad)
    return updates

def gd_momentum(params, grads, lr, mc):  #gradient decent with momentum
    updates = {}
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        param_prev = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        updates[param] = param - (lr * grad) - mc*param_prev  
        #or updates[param] = param - (lr * (1 - mc) *  grad) - mc*param_prev
        #https://www.mathworks.com/help/nnet/ref/traingdm.html
        updates[param_prev] = param
    return updates
