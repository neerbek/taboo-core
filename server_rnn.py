# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:11:21 2017

@author: neerbek
"""


import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T
from datetime import datetime

import similarity.load_trees as load_trees

import deeplearning_tutorial.rnn4 as nn_model
import rnn_enron
from ai_util import TrainTimer

class State:
    def __init__(self):
        self.nx = None
        self.LT = None
        self.train_trees = None
        self.dev_trees = None
        self.test_trees = None
        self.is_initialized = False
state = State()   #global state

def setWordSize(wordSize):
    state.nx = wordSize
    rnn_enron.Evaluator.set_size(state.nx)

def initialize_state(max_count=-1):
    if not state.is_initialized:
        setWordSize(50)
        rng = RandomState(1234)
        state.LT = rnn_enron.get_word_embeddings("../code/glove/glove.6B.{}d.txt".format(state.nx), rng, max_count)
        rnn_enron.Ctxt.evaltimer = TrainTimer("eval timer")
        rnn_enron.Ctxt.appendtimer = TrainTimer("append timer")
        state.is_initialized = True

def load_model(trainer, max_count=-1):
    state.train_trees = load_trees.get_trees('trees/train.txt', max_count)
    state.valid_trees = load_trees.get_trees('trees/dev.txt', max_count)
    state.test_trees = load_trees.get_trees('trees/test.txt', max_count)
    rnn_enron.initializeTrees(state.train_trees, state.LT)
    rnn_enron.initializeTrees(state.valid_trees, state.LT)
    rnn_enron.initializeTrees(state.test_trees, state.LT)
    trainer.update_batch_size()
        

class Trainer:
    def __init__(self):
        self.learning_rate=0.01
        self.L1_reg=0.00
        self.L2_reg=0.0001
        self.n_epochs=1000
        self.batch_size=40
        self.retain_probability = 0.8
        self.n_train_batches = 0
        self.valid_batch_size = 0
        self.n_valid_batches = 0
        self.n_test_batches = 0

    def update_batch_size(self):
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = len(state.train_trees) // self.batch_size
        self.valid_batch_size = len(state.valid_trees)
        self.n_valid_batches = len(state.valid_trees) // self.valid_batch_size
        self.n_test_batches = len(state.test_trees) // self.batch_size
        
    def train(self, rnn, n_epochs=1, rng=RandomState(1234)):
        validation_frequency = 1   #self.n_train_batches/2
        epoch = 0
        it = 0
        batch_size = self.batch_size
        reg = rnn.rnn
        cost = reg.cost(rnn.y) + self.L1_reg * reg.L1 + self.L2_reg * reg.L2_sqr
            
        validate_model = theano.function(
            inputs=[rnn.x,rnn.y,rnn.z],
            outputs=reg.errors(rnn.y)
        )
        
        gparams = [T.grad(cost, param) for param in reg.params]
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(reg.params, gparams)
        ]
        
        cost_model = theano.function(
            inputs=[rnn.x,rnn.y,rnn.z],
            outputs=cost
        )
        
        train_model = theano.function(
            inputs=[rnn.x,rnn.y,rnn.z],
            outputs=cost,
            updates=updates
        )
        while (n_epochs==-1 or epoch < n_epochs):
            epoch += 1
            minibatch_index=0
            for minibatch_index in range(self.n_train_batches):
                trees = state.train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                evaluator = rnn_enron.Evaluator(reg)
                (roots, x_val, y_val) = rnn_enron.getInputArrays(reg, trees, evaluator)
                minibatch_avg_cost = []
                z_val = rng.binomial(n=1, size=(x_val.shape[0], rnn_enron.Evaluator.HIDDEN_SIZE), p=self.retain_probability)
                minibatch_avg_cost.append(train_model(x_val, y_val, z_val))
                minibatch_avg_cost = numpy.mean(minibatch_avg_cost)
                it += 1
                if it % validation_frequency == 0:
                    validation_losses = []
                    val_total_zeros = []
                    val_root_losses = []
                    val_root_zeros = []
                    validation_cost = []
                    evaluator = rnn_enron.Evaluator(reg)
                    #c = prng.randint(n_valid_batches, size=n_valid_batches/train_times)
                    for i in range(self.n_valid_batches):
                        trees = state.valid_trees[i * self.valid_batch_size: (i + 1) * self.valid_batch_size]
                        (roots, x_val, y_val) = rnn_enron.getInputArrays(reg, trees, evaluator)
                        z_val = numpy.ones(shape=(x_val.shape[0], rnn_enron.Evaluator.HIDDEN_SIZE))
                        z_val = z_val * self.retain_probability
                        validation_losses.append(validate_model(x_val, y_val, z_val))
                        validation_cost.append(cost_model(x_val, y_val, z_val))
                        val_total_zeros.append(rnn_enron.get_zeros(y_val))
                        x_roots = []
                        y_roots = []
                        for r in roots:
                            x_roots.append(x_val[r,:])
                            y_roots.append(y_val[r,:])
                        z_roots = numpy.ones(shape=(len(x_roots), rnn_enron.Evaluator.HIDDEN_SIZE))
                        z_roots = z_roots * self.retain_probability
                        val_root_losses.append(validate_model(x_roots, y_roots, z_roots))
                        val_root_zeros.append(rnn_enron.get_zeros(y_roots))
                    val_total_acc = 1 - numpy.mean(validation_losses)
                    val_root_acc = 1  - numpy.mean(val_root_losses)
                    val_total_zeros = 1 - numpy.mean(val_total_zeros)
                    val_root_zeros = 1 - numpy.mean(val_root_zeros)
                    val_cost = numpy.mean(validation_cost)
                    print("epoch {}. time is {}, minibatch {}/{}, validation total accuracy {:.4f} % ({:.4f} %) validation cost {:.6f}, val root acc {:.4f} % ({:.4f} %)".format(
                            epoch, datetime.now().strftime('%d-%m %H:%M'),
                            minibatch_index + 1, self.n_train_batches, val_total_acc*100., val_total_zeros*100.,
                            val_cost*1.0, val_root_acc*100., val_root_zeros*100.
                        ))
                    #have to mult by 1.0 to convert minibatch_avg_cost from theano to python variables


class RNNWrapper:
    def __init__(self):
        self.x = T.matrix('x')  
        self.y = T.matrix('y')  
        self.z = T.matrix('z')    #for dropout
        rng = RandomState(1234)
        # Define RNN
        self.rnn = nn_model.RNN(rng=rng, X=self.x, Z=self.z, n_in=2*(rnn_enron.Evaluator.SIZE+rnn_enron.Evaluator.HIDDEN_SIZE), 
              n_hidden=rnn_enron.Evaluator.HIDDEN_SIZE, n_out=rnn_enron.Evaluator.RES_SIZE
              )
        self.y_pred = self.rnn.y_pred
        
        self.run_model = theano.function(
            inputs=[self.x,self.z],
            outputs=self.y_pred
            )


    def load(self, filename='model.save'):
        nn_model.load(self.rnn, filename)


#train = load_trees.get_trees('trees/train.txt')
#rnn_enron.initializeTrees(train, state.LT)
#train_trees = train[100:200]
#(list_root_indexes, x_val, y_val) = rnn_enron.getInputArrays(rnn.rnn, train_trees, evaluator)
def get_predictions(rnn, indexed_sentences):
    evaluator = rnn_enron.Evaluator(rnn.rnn)
    trees = []
    for s in indexed_sentences:
        trees.append(s.tree)
    (list_root_indexes, x_val, y_val) = rnn_enron.getInputArrays(rnn.rnn, trees, evaluator)
    
    x_roots = []
    y_roots = []
    for r in list_root_indexes:
        x_roots.append(x_val[r,:])
        y_roots.append(y_val[r,:])
    z_roots = numpy.ones(shape=(len(x_roots), rnn_enron.Evaluator.HIDDEN_SIZE))
    z_roots = z_roots * 0.9
    pred = rnn.run_model(x_roots, z_roots)
    for i in range(len(pred)):
        indexed_sentences[i].pred = pred[i]

if __name__ == "__main__":
    #testing
    import server_rnn_helper
    setWordSize(50)
    initialize_state()
    rnn = RNNWrapper()
    rnn.load('model.save')
    
    text = """Please have a look at enclosed worksheets.
    As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact. This may impact the final decision.
    We may have to move to cash margining if necessary."""
    indexed_sentences = server_rnn_helper.get_indexed_sentences(text)
    trees = server_rnn_helper.get_nltk_trees(0, indexed_sentences)
    rnn_enron.initializeTrees(trees, state.LT)    
    get_predictions(rnn, indexed_sentences)
    
    for s in indexed_sentences:
        print(s.pred)
