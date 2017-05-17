# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:11:21 2017

@author: neerbek
"""

import os
#import psutil
import numpy
from numpy.random import RandomState
import theano
import theano.tensor as T
from datetime import datetime
import math
import objgraph

import similarity.load_trees as load_trees

import deeplearning_tutorial.rnn4 as nn_model
import rnn_enron
from ai_util import Timer

DEBUG_PRINT = True
class State:
    def __init__(self, max_embedding_count=-1, nx=50, nh=300, rng=RandomState(1234), glove_path="../code/glove/"):
        self.nx = None
        self.LT = None
        self.train_trees = None
        self.valid_trees = None
        self.test_trees = None
        self.setWordSize(nx, nh)
        self.LT = rnn_enron.get_word_embeddings(os.path.join(glove_path, "glove.6B.{}d.txt".format(self.nx)), rng, max_embedding_count)
        rnn_enron.Ctxt.evaltimer = Timer("eval timer")
        rnn_enron.Ctxt.appendtimer = Timer("append timer")

    def setWordSize(self, wordSize, hiddenSize):
        self.nx = wordSize
        rnn_enron.Evaluator.set_size(self.nx, hiddenSize)

    def load_trees(self, trainer, max_tree_count=-1):
        self.train_trees = load_trees.get_trees('trees/train.txt', max_tree_count)
        self.valid_trees = load_trees.get_trees('trees/dev.txt', max_tree_count)
        self.test_trees = load_trees.get_trees('trees/test.txt', max_tree_count)
        self.init_trees(trainer)        
    
    def init_trees(self, trainer):
        rnn_enron.initializeTrees(self.train_trees, self.LT)
        rnn_enron.initializeTrees(self.valid_trees, self.LT)
        rnn_enron.initializeTrees(self.test_trees, self.LT)
        trainer.update_batch_size(self)

class PerformanceMeasurer:
    def __init__(self, performanceMeasurer=None):
        if performanceMeasurer==None:
            self.total_acc = 0
            self.root_acc = 0
            self.total_zeros = 0
            self.root_zeros = 0
            self.cost = 0
        else:
            self.total_acc = performanceMeasurer.total_acc
            self.root_acc = performanceMeasurer.root_acc
            self.total_zeros = performanceMeasurer.total_zeros
            self.root_zeros = performanceMeasurer.root_zeros
            self.cost = performanceMeasurer.cost
        
    def measure(me, state, trainer, rnn, validate_model, cost_model):
        validation_losses = []
        val_total_zeros = []
        val_root_losses = []
        val_root_zeros = []
        validation_cost = []
        evaluator = rnn_enron.Evaluator(rnn)
        #c = prng.randint(n_valid_batches, size=n_valid_batches/train_times)
        for i in range(trainer.n_valid_batches):
            trees = state.valid_trees[i * trainer.valid_batch_size: (i + 1) * trainer.valid_batch_size]
            (roots, x_val, y_val) = rnn_enron.getInputArrays(rnn, trees, evaluator)
            z_val = numpy.ones(shape=(x_val.shape[0], rnn.n_hidden))
            z_val = z_val * trainer.retain_probability
            z_val = z_val.astype(dtype=theano.config.floatX)
            validation_losses.append(validate_model(x_val, y_val, z_val))
            validation_cost.append(cost_model(x_val, y_val, z_val))
            val_total_zeros.append(rnn_enron.get_zeros(y_val))
            x_roots = []
            y_roots = []
            for r in roots:
                x_roots.append(x_val[r,:])
                y_roots.append(y_val[r,:])
            z_roots = trainer.retain_probability*numpy.ones(shape=(len(x_roots), rnn.n_hidden))
            z_roots = z_roots.astype(dtype=theano.config.floatX)
            val_root_losses.append(validate_model(x_roots, y_roots, z_roots))
            val_root_zeros.append(rnn_enron.get_zeros(y_roots))
        me.total_acc = 1 - numpy.mean(validation_losses)
        me.root_acc = 1  - numpy.mean(val_root_losses)
        me.total_zeros = 1 - numpy.mean(val_total_zeros)
        me.root_zeros = 1 - numpy.mean(val_root_zeros)
        me.cost = numpy.mean(validation_cost)
        
    def measure_trees(self, trees, retain_probability, rnn, validation_model, cost_model):
        validation_losses = 0
        val_total_zeros = 0
        val_root_losses = 0
        val_root_zeros = 0
        validation_cost = 0
        evaluator = rnn_enron.Evaluator(rnn)
        (roots, x_val, y_val) = rnn_enron.getInputArrays(rnn, trees, evaluator)
        z_val = numpy.ones(shape=(x_val.shape[0], rnn.n_hidden))
        z_val = z_val * retain_probability
        z_val = z_val.astype(dtype=theano.config.floatX)
        validation_losses = validation_model(x_val, y_val, z_val)
        validation_cost = cost_model(x_val, y_val, z_val)
        val_total_zeros = rnn_enron.get_zeros(y_val)
        x_roots = []
        y_roots = []
        for r in roots:
            x_roots.append(x_val[r,:])
            y_roots.append(y_val[r,:])
        z_roots = retain_probability*numpy.ones(shape=(len(x_roots), rnn.n_hidden))
        val_root_losses = validation_model(x_roots, y_roots, z_roots)
        val_root_zeros = rnn_enron.get_zeros(y_roots)
        self.total_acc = 1 - validation_losses
        self.root_acc = 1  - val_root_losses
        self.total_zeros = 1 - val_total_zeros
        self.root_zeros = 1 - val_root_zeros
        self.cost = validation_cost
        
    def report(self, msg = ""):
        print(msg + " total accuracy {:.4f} % ({:.4f} %) cost {:.6f}, root acc {:.4f} % ({:.4f} %)".format(self.total_acc*100., 
              self.total_zeros*100., self.cost*1.0, 
              self.root_acc*100., self.root_zeros*100.
              ))


class Trainer:
    def __init__(self, trainer=None):
        if trainer!=None:
            self.learning_rate=trainer.learning_rate
            self.L1_reg=trainer.L1_reg
            self.L2_reg=trainer.L2_reg
            self.n_epochs=trainer.n_epochs
            self.batch_size=trainer.batch_size
            self.retain_probability = trainer.retain_probability
            self.n_train_batches = trainer.n_train_batches
            self.valid_batch_size = trainer.valid_batch_size
            self.n_valid_batches = trainer.n_valid_batches
            self.n_test_batches = trainer.n_test_batches
        else:
            self.learning_rate=0.01
            self.L1_reg=0.0
            self.L2_reg=0.0001
            self.n_epochs=1000
            self.batch_size=40
            self.retain_probability = 0.8
            self.n_train_batches = 0
            self.valid_batch_size = 0
            self.n_valid_batches = 0
            self.n_test_batches = 0

    def update_batch_size(self, state):
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = int(math.ceil(len(state.train_trees) / self.batch_size))
        self.valid_batch_size = len(state.valid_trees)
        self.n_valid_batches = int(math.ceil(len(state.valid_trees) / self.valid_batch_size))
        self.n_test_batches = int(math.ceil(len(state.test_trees) / self.batch_size))
        
    def get_cost(self, rnnWrapper):
        reg = rnnWrapper.rnn
        return reg.cost(rnnWrapper.y) + self.L1_reg * reg.L1 + self.L2_reg * reg.L2_sqr

    def get_cost_model(self, rnnWrapper, cost):
        cost_model = theano.function(
            inputs=[rnnWrapper.x,rnnWrapper.y,rnnWrapper.z],
            outputs=cost
        )
        return cost_model

    def get_validation_model(self, rnnWrapper):
        return theano.function(
            inputs=[rnnWrapper.x,rnnWrapper.y,rnnWrapper.z],
            outputs=rnnWrapper.rnn.errors(rnnWrapper.y)
        )
    def evaluate_model(self, trees, rnnWrapper, validation_model, cost_model):
        performanceMeasurer = PerformanceMeasurer()
        performanceMeasurer.measure_trees(trees=trees, retain_probability = self.retain_probability, 
                                          rnn = rnnWrapper.rnn, validation_model=validation_model, 
                                          cost_model=cost_model)
        return performanceMeasurer
        
    def train(self, state, rnnWrapper, file_prefix="save", n_epochs=1, rng=RandomState(1234), epoch=0, validation_frequency=1, train_report_frequency=1, balance_trees=False):
        it = 0
        batch_size = self.batch_size
        reg = rnnWrapper.rnn
        cost = self.get_cost(rnnWrapper)
            
        validate_model = self.get_validation_model(rnnWrapper)
        
        gparams = [T.grad(cost, param) for param in reg.params]
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(reg.params, gparams)
        ]
        
        cost_model = self.get_cost_model(rnnWrapper, cost)
        
        train_model = theano.function(
            inputs=[rnnWrapper.x,rnnWrapper.y,rnnWrapper.z],
            outputs=cost,
            updates=updates
        )
        performanceMeasurerBest = PerformanceMeasurer()
        performanceMeasurerBest.epoch=-1
        performanceMeasurer = PerformanceMeasurer()
        performanceMeasurer.epoch = -1
        while (n_epochs==-1 or epoch < n_epochs):
            perm = rng.permutation(len(state.train_trees))
            train_trees =  [state.train_trees[i] for i in perm]
            epoch += 1
            for minibatch_index in range(self.n_train_batches):
                objgraph.show_most_common_types()
                trees = train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                if balance_trees:
                    trees = get_balanced_data(trees, rng, state)
                if len(trees)==0:
                    print("continueing")
                    continue
                evaluator = rnn_enron.Evaluator(reg)
                (roots, x_val, y_val) = rnn_enron.getInputArrays(reg, trees, evaluator)
                z_val = rng.binomial(n=1, size=(x_val.shape[0], rnn_enron.Evaluator.HIDDEN_SIZE), p=self.retain_probability)
                z_val = z_val.astype(dtype=theano.config.floatX)
                minibatch_cost = train_model(x_val, y_val, z_val)
                it += 1
                if it % train_report_frequency == 0:
                    if DEBUG_PRINT:
                        #process = psutil.Process(os.getpid())
                        #print(str(process.memory_info().rss/1000000) + " MB")                        
                        minibatch_acc = 1 - validate_model(x_val, y_val, z_val)
                        minibatch_zeros = 1 - rnn_enron.get_zeros(y_val)
                        print("epoch {}. time is {}, minibatch {}/{}, On train set: cost {:.6f} batch acc {:.4f} %  ({:.4f} %)".format(epoch, datetime.now().strftime('%d-%m %H:%M'), minibatch_index + 1, self.n_train_batches, minibatch_cost*1.0, minibatch_acc*100.0, minibatch_zeros*100.0
                        ))
                if it % validation_frequency == 0:
                    performanceMeasurer = PerformanceMeasurer()
                    performanceMeasurer.epoch = epoch
                    performanceMeasurer.measure(state, self,  reg, validate_model, cost_model)
                    if DEBUG_PRINT:
                        #process = psutil.Process(os.getpid())
                        #print(str(process.memory_info().rss/1000000) + " MB")                        
                        performanceMeasurer.report(msg = "epoch {}. time is {}, minibatch {}/{}, On validation set:".format(epoch, 
                                                   datetime.now().strftime('%d-%m %H:%M'), minibatch_index + 1, 
                                self.n_train_batches))
                    if performanceMeasurerBest.root_acc<performanceMeasurer.root_acc:
                        filename = "{}_best.txt".format(file_prefix)
                        self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                        performanceMeasurerBest = performanceMeasurer
                    else:
                        if performanceMeasurerBest.epoch + 2 < epoch:
                            filename = "{}_running.txt".format(file_prefix)
                            self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                            performanceMeasurerBest.epoch = epoch
        filename = "{}_running.txt".format(file_prefix)
        self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
        
    def save(self, rnnWrapper, filename, epoch, performanceMeasurer, performanceMeasurerBest):
        print("Saving rnnWrapper. Previous {};{:.4f}. New {};{:.4f}".format(performanceMeasurerBest.epoch, performanceMeasurerBest.root_acc, performanceMeasurer.epoch, performanceMeasurer.root_acc))
        print("Saving as " + filename)
        rnnWrapper.save(filename, epoch, performanceMeasurer.root_acc)

class RNNWrapper:
    def __init__(self, rng = RandomState(1234)):
        self.x = T.matrix('x', dtype=theano.config.floatX)  
        self.y = T.matrix('y', dtype=theano.config.floatX)  
        self.z = T.matrix('z', dtype=theano.config.floatX)    #for dropout
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
        return nn_model.load(rnn=self.rnn, filename=filename)

    def save(self, filename='model.save', epoch=0, acc=0):
        nn_model.save(rnn=self.rnn, filename=filename, epoch=epoch, acc=acc)

class IteratorGuard:
    def __init__(self):
        self.a = "a"

def get_balanced_data(trees, rng, state = None):
    zero_trees = []
    four_trees = []
    count = 0
    try:
        for t in trees:
            if t.syntax=='0':
                zero_trees.append(t)
            elif t.syntax=='4':
                four_trees.append(t)
            count += 1
    except AttributeError:
        print("attribute error in loop len(trees) {}, count {}, type trees {}, type t {} type t {}".format(len(trees), count, type(trees), type(t), type(t)))
        print("type next(t) {}".format(type(next(t, IteratorGuard()))))
        raise
    if len(zero_trees)+ len(four_trees)!=len(trees):
        raise Exception("expected lengths to match {} + {} == {}".format(len(zero_trees),len(four_trees),len(trees))
        )
    min_list = zero_trees
    max_list = four_trees
    if (len(zero_trees)>len(four_trees)):
        min_list, max_list = four_trees, zero_trees
    if len(min_list)<2:
        print("for training no examples in mimimum list")
        return []
    
    length = int(len(min_list)*1)
    length = min(length, len(max_list))
    choices = rng.choice(len(max_list), size=length, replace=False)
    max_list = [max_list[i] for i in choices]
    if len(max_list)<3:
        print("after pruning only: {} elements in list".format(2*len(max_list)))
        return []
    res = min_list
    res.extend(max_list)
    #print("len(res)", len(res))
    perm = rng.permutation(len(res))
    res = [res[i] for i in perm]
    return res
        
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
    z_roots = z_roots.astype(dtype=theano.config.floatX)

    pred = rnn.run_model(x_roots, z_roots)
    for i in range(len(pred)):
        indexed_sentences[i].pred = pred[i]

if __name__ == "__main__":
    #testing
    import server_rnn_helper
    state = State()
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
