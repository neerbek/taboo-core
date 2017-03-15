# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:03:20 2016

@author: neerbek
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

__file__ = '/home/neerbek/jan/phd/DLP/paraphrase/python/deeplearning_tutorial'

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
        #TODO: test this works with matrix y
        
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

    
import logistic_sgd
dataset='mnist.pkl.gz'
batch_size=600
datasets = logistic_sgd.load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

def transform_y(y_in):
    y = y_in.eval()   #get numpy arrayb (I think)
    arr = numpy.zeros((y.shape[0],10), dtype=theano.config.floatX)
    for i in range(y.shape[0]):
        arr[i,y[i]] = 1.0
    shared_y = theano.shared(arr, borrow=True)
    return shared_y

#train_set_x.get_value(borrow=True).shape[0]    
#y_in = train_set_y
train_set_y = transform_y(train_set_y)
valid_set_y = transform_y(valid_set_y)
test_set_y = transform_y(test_set_y)


######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.matrix('y')  # labels, presented as one-hot matrix of labels

# Each MNIST image has size 28*28
reg = Regression(X=x, n_in=28 * 28, n_out=10)

cost = reg.cost(y)

# compiling a Theano function that computes the mistakes that are made by
# the model on a minibatch
test_model = theano.function(
    inputs=[index],
    outputs=reg.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=reg.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# compute the gradient of cost with respect to theta = (W,b)
g_W = T.grad(cost=cost, wrt=reg.W)
g_b = T.grad(cost=cost, wrt=reg.b)

learning_rate=0.13
updates = [(reg.W, reg.W - learning_rate * g_W),
           (reg.b, reg.b - learning_rate * g_b)]

# compiling a Theano function `train_model` that returns the cost, but in
# the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
    
validation_frequency = n_train_batches

n_epochs=1000
epoch = 0
while (epoch < n_epochs):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        it = (epoch - 1) * n_train_batches + minibatch_index
        if (it + 1) % validation_frequency == 0:
            validation_losses = [validate_model(i)
                                 for i in range(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %% cost %f' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.,
                    minibatch_avg_cost
                )
            )
cost.eval({X:train_set_x.get_value(),y:train_set_y.get_value()})
#end_time = timeit.default_timer()
#print(
#    (
#        'Optimization complete with best validation score of %f %%,'
#        'with test performance %f %%'
#    )
#    % (best_validation_loss * 100., test_score * 100.)
#)
#print('The code run for %d epochs, with %f epochs/sec' % (
#    epoch, 1. * epoch / (end_time - start_time)))
#print(('The code for file ' +
#       os.path.split(__file__)[1] +
#       ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
#
#
