# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:11:21 2017

@author: neerbek
"""

import os
# import psutil
import numpy                          # type: ignore
from numpy.random import RandomState  # type: ignore
import theano                         # type: ignore
import theano.tensor as T             # type: ignore
from datetime import datetime
import math
# import gc

import similarity.load_trees as load_trees

import rnn_model.rnn as nn_model
import rnn_model.learn
import rnn_enron
import ai_util
import LogFileReader

DEBUG_PRINT = True
class State:
    def __init__(self, max_embedding_count=-1, nx=50, nh=300, rng=RandomState(1234), glove_path="../code/glove/") -> None:
        self.nx = None
        self.LT = None
        self.train_trees = None
        self.valid_trees = None
        self.test_trees = None
        self.setWordSize(nx, nh)
        self.LT = rnn_enron.get_word_embeddings(os.path.join(glove_path, "glove.6B.{}d.txt".format(self.nx)), rng, max_embedding_count)
        rnn_enron.RNNTimers.init()  # reset timers

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
class Timers:
    totaltimer = ai_util.Timer("measure_trees")
    randomtimer = ai_util.Timer("random")
    calltheanotimer = ai_util.Timer("callTheano")

# for when we don't want to calc confusion matrix
def empty_confusion_matrix(x, y, z):
    return (-1, -1, -1, -1)

class PerformanceMeasurer:
    def __init__(self, performanceMeasurer=None):
        if performanceMeasurer == None:
            self.total_acc = 0
            self.root_acc = 0
            self.total_zeros = 0
            self.root_zeros = 0
            self.cost = 0
            self.epoch = -1
            self.running_epoch = -1
        else:
            self.total_acc = performanceMeasurer.total_acc
            self.root_acc = performanceMeasurer.root_acc
            self.total_zeros = performanceMeasurer.total_zeros
            self.root_zeros = performanceMeasurer.root_zeros
            self.cost = performanceMeasurer.cost
            self.epoch = performanceMeasurer.epoch
            self.running_epoch = performanceMeasurer.running_epoch

    def measure(self, state, trainer, rnn, validate_model, cost_model, confusion_matrix=empty_confusion_matrix):
        self.measure_trees(input_trees=state.valid_trees, batch_size=trainer.valid_batch_size,
                           retain_probability=trainer.retain_probability, rnn=rnn,
                           validate_model=validate_model, cost_model=cost_model, confusion_matrix=confusion_matrix)

    def measure_trees(self, input_trees, batch_size, retain_probability, rnn, validate_model, cost_model, confusion_matrix=empty_confusion_matrix):
        # Timers.totaltimer.begin()
        validation_losses = 0  # errors
        validation_cost = 0  # cost
        validation_zeros = 0  # number of non-sensitive
        validation_confusion_matrix = [0, 0, 0, 0]
        val_root_losses = 0
        val_root_zeros = 0
        val_root_confusion_matrix = [0, 0, 0, 0]
        total_nodes = 0
        total_root_nodes = 0
        evaluator = rnn_enron.Evaluator(rnn)
        n_batches = int(math.ceil(len(input_trees) / batch_size))
        for i in range(n_batches):
            trees = input_trees[i * batch_size: (i + 1) * batch_size]
            (roots, x_val, y_val) = rnn_enron.getInputArrays(rnn, trees, evaluator)
            z_val = retain_probability * numpy.ones(shape=(x_val.shape[0], rnn.n_hidden), dtype=theano.config.floatX)
            n_nodes = x_val.shape[0]
            validation_losses += validate_model(x_val, y_val, z_val) * n_nodes  # append mean fraction of errors
            validation_cost += cost_model(x_val, y_val, z_val) * n_nodes        # append mean cost
            validation_zeros += rnn_enron.get_zeros(y_val) * n_nodes
            a = confusion_matrix(x_val, y_val, z_val)
            for i in range(len(validation_confusion_matrix)):
                validation_confusion_matrix[i] += a[i]  # a is list of arrays (theano is weird)
            total_nodes += n_nodes
            x_roots = []
            y_roots = []
            for r in roots:
                x_roots.append(x_val[r, :])
                y_roots.append(y_val[r, :])
            z_roots = retain_probability * numpy.ones(shape=(len(x_roots), rnn.n_hidden), dtype=theano.config.floatX)
            n_root_nodes = len(x_roots)
            val_root_losses += validate_model(x_roots, y_roots, z_roots) * n_root_nodes
            val_root_zeros += rnn_enron.get_zeros(y_roots) * n_root_nodes
            a = confusion_matrix(x_roots, y_roots, z_roots)
            for i in range(len(val_root_confusion_matrix)):
                val_root_confusion_matrix[i] += a[i]  # a is list of arrays (theano is weird)
            total_root_nodes += n_root_nodes
        self.total_acc = 1 - validation_losses / total_nodes
        self.root_acc = 1 - val_root_losses / total_root_nodes
        self.total_zeros = 1 - validation_zeros / total_nodes
        self.total_nodes = total_nodes
        self.root_zeros = 1 - val_root_zeros / total_root_nodes
        self.cost = validation_cost / total_nodes
        self.total_root_nodes = total_root_nodes
        self.total_confusion_matrix = validation_confusion_matrix
        self.root_confusion_matrix = val_root_confusion_matrix
        # Timers.totaltimer.end()

    def measure_roots(self, input_trees, batch_size, retain_probability, rnn, measure_wrapper, measureRoots=True):
        "Measure wrapper/function here receives x, y and z values and also the list of trees. This allow the wrapper of doing corpus specific measures"
        print("measure_roots: ", measureRoots)
        evaluator = rnn_enron.Evaluator(rnn)
        n_batches = int(math.ceil(len(input_trees) / batch_size))
        for i in range(n_batches):
            trees = input_trees[i * batch_size: (i + 1) * batch_size]
            (roots, x_val, y_val) = rnn_enron.getInputArrays(rnn, trees, evaluator)
            x_roots = []
            y_roots = []
            if measureRoots:
                for r in roots:
                    x_roots.append(x_val[r, :])
                    y_roots.append(y_val[r, :])
            else:  # do all
                prevRootIndex = 0
                newTrees = []
                rootIndex = 0
                # print("measure_roots: ", rootIndex, roots[rootIndex], 0, roots[rootIndex] - 0 + 1)
                addCount = roots[rootIndex] - 0 + 1
                tmp = [trees[0] for add in range(addCount)]
                newTrees.extend(tmp)

                for rootIndex in range(1, len(roots)):
                    # print("measure_roots: ", rootIndex, roots[rootIndex], roots[prevRootIndex], roots[rootIndex] - roots[prevRootIndex])
                    addCount = roots[rootIndex] - roots[prevRootIndex]
                    tmp = [trees[rootIndex] for add in range(addCount)]
                    newTrees.extend(tmp)
                    prevRootIndex = rootIndex
                # print("measure_roots: ", rootIndex, len(x_val), roots[prevRootIndex], len(x_val) - roots[prevRootIndex])
                # addCount = len(x_val) - roots[prevRootIndex]
                # tmp = [trees[prevRootIndex] for add in range(addCount)]
                # newTrees.extend(tmp)

                # print("measure_roots: ", len(trees))
                trees = newTrees
                x_roots = x_val
                y_roots = y_val
                # print("measure_roots: ", len(x_roots), len(trees))
            z_roots = retain_probability * numpy.ones(shape=(len(x_roots), rnn.n_hidden), dtype=theano.config.floatX)
            measure_wrapper(x_roots, y_roots, z_roots, trees)

class Trainer:
    def __init__(self, trainer=None):
        if trainer != None:
            self.learning_rate = trainer.learning_rate
            self.mc = trainer.learning_rate
            self.L1_reg = trainer.L1_reg
            self.L2_reg = trainer.L2_reg
            self.n_epochs = trainer.n_epochs
            self.batch_size = trainer.batch_size
            self.retain_probability = trainer.retain_probability
            self.n_train_batches = trainer.n_train_batches
            self.valid_batch_size = trainer.valid_batch_size
            self.n_valid_batches = trainer.n_valid_batches
            self.n_test_batches = trainer.n_test_batches
            self.cm = trainer.cm
            self.learn = trainer.learn
        else:
            self.learning_rate = 0.01
            self.mc = 0
            self.L1_reg = 0.0
            self.L2_reg = 0.0001
            self.n_epochs = 1000
            self.batch_size = 40
            self.retain_probability = 0.8
            self.n_train_batches = 0
            self.valid_batch_size = 0
            self.n_valid_batches = 0
            self.n_test_batches = 0
            self.cm = 0
            self.learn = "gd"

    def update_batch_size(self, state):
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = int(math.ceil(len(state.train_trees) / self.batch_size))
        if self.valid_batch_size == 0:
            self.valid_batch_size = len(state.valid_trees)
        self.n_valid_batches = int(math.ceil(len(state.valid_trees) / self.valid_batch_size))
        self.n_test_batches = int(math.ceil(len(state.test_trees) / self.batch_size))

    def get_cost(self, rnnWrapper):
        reg = rnnWrapper.rnn
        return reg.cost(rnnWrapper.y) + self.L1_reg * reg.L1 + self.L2_reg * reg.L2_sqr

    def get_cost_model(self, rnnWrapper, cost):
        cost_model = theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=cost
        )
        return cost_model

    def get_confusion_matrix(self, rnnWrapper):
        # this is a function in order to be accessable from tests
        return theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=rnnWrapper.rnn.regressionLayer.confusion_matrix(rnnWrapper.y)
        )

    def get_validation_model(self, rnnWrapper):
        # this is a function in order to be accessable from tests
        return theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=rnnWrapper.rnn.errors(rnnWrapper.y)
        )

    def get_updates(self, params, grads):
        if self.learn == 'gd':
            return rnn_model.learn.gd(params=params, grads=grads, lr=self.learning_rate)
        elif self.learn == 'gdm':
            return rnn_model.learn.gd_momentum(params=params, grads=grads, lr=self.learning_rate, mc=self.mc)
        elif self.learn == 'adagrad':
            return rnn_model.learn.adagrad(params=params, grads=grads, lr=self.learning_rate)
        else:
            raise Exception("unknown learner: " + self.learn)

    def evaluate_model(self, trees, rnnWrapper, validation_model, cost_model):
        performanceMeasurer = PerformanceMeasurer()
        performanceMeasurer.measure_trees(input_trees=trees, batch_size=self.valid_batch_size,
                                          retain_probability=self.retain_probability,
                                          rnn=rnnWrapper.rnn, validate_model=validation_model,
                                          cost_model=cost_model)
        return performanceMeasurer

    def train(self, state, rnnWrapper, file_prefix="save", n_epochs=1, rng=RandomState(1234), epoch=0, validation_frequency=1, train_report_frequency=1, output_running_model=-1, balance_trees=False):
        it = 0
        batch_size = self.batch_size
        reg = rnnWrapper.rnn
        cost = self.get_cost(rnnWrapper)
        vali = rnnWrapper.rnn.errors(rnnWrapper.y)

        validate_model = self.get_validation_model(rnnWrapper)
        cost_model = self.get_cost_model(rnnWrapper, cost)
        confusion_matrix = self.get_confusion_matrix(rnnWrapper)

        params = reg.params
        grads = [T.grad(cost=cost, wrt=param) for param in params]
        updates = self.get_updates(params=params, grads=grads)

        update_keys = [k for k in updates.keys()]

        # mem leak:
        # reported: https://github.com/Theano/Theano/issues/5810
        # fixed: https://github.com/Theano/Theano/pull/5832

        for k in update_keys:
            print("key:" + str(k.name))
        # for k in update_keys2:
        #     print("key2:" + str(k.name))

        outputs = [vali, cost] + [updates[k] for k in update_keys]

        train = theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=outputs)

        performanceMeasurerBest = PerformanceMeasurer()
        performanceMeasurerBest.epoch = -1
        performanceMeasurerBest.running_epoch = -1
        performanceMeasurer = PerformanceMeasurer()
        performanceMeasurer.epoch = -1

        while (n_epochs == -1 or epoch < n_epochs):
            # Timers.randomtimer.begin()
            perm = rng.permutation(len(state.train_trees))
            # Timers.randomtimer.end()
            train_trees = [state.train_trees[i] for i in perm]
            epoch += 1
            train_cost = 0
            train_acc = 0
            train_count = 0
            for minibatch_index in range(self.n_train_batches):
                trees = train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                if balance_trees:
                    trees = get_balanced_data(trees, rng, state)
                if len(trees) == 0:
                    print("continueing")
                    continue
                evaluator = rnn_enron.Evaluator(reg)
                (roots, x_val, y_val) = rnn_enron.getInputArrays(reg, trees, evaluator)
                z_val = rng.binomial(n=1, size=(x_val.shape[0], rnn_enron.Evaluator.HIDDEN_SIZE), p=self.retain_probability)
                z_val = z_val.astype(dtype=theano.config.floatX)
                # Timers.calltheanotimer.begin()
                values = train(x_val, y_val, z_val)
                train_acc += (1 - values[0]) * x_val.shape[0]
                train_cost += values[1] * x_val.shape[0]
                train_count += x_val.shape[0]
                for index, param in enumerate(update_keys):
                    param.set_value(values[index + 2])
                # Timers.calltheanotimer.end()
                it += 1
                if it % train_report_frequency == 0:
                    if DEBUG_PRINT:
                        minibatch_zeros = 1 - rnn_enron.get_zeros(y_val)
                        minibatch_acc = 1 - values[0]  # validate_model(x_val, y_val, z_val)
                        print("epoch {}. time is {}, minibatch {}/{}, On train set: batch acc {:.4f} %  ({:.4f} %)".format(epoch, datetime.now().strftime('%d-%m %H:%M'), minibatch_index + 1, self.n_train_batches, minibatch_acc * 100.0, minibatch_zeros * 100.0))
                        LogFileReader.logTrain(LogFileReader.Train(cost=train_cost / train_count, nodeAccuracy=train_acc / train_count, nodeCount=train_count), epoch=epoch)
                if it % validation_frequency == 0:
                    performanceMeasurer = PerformanceMeasurer()
                    performanceMeasurer.epoch = epoch
                    performanceMeasurer.measure(state=state, trainer=self, rnn=reg, validate_model=validate_model, cost_model=cost_model, confusion_matrix=confusion_matrix)
                    if DEBUG_PRINT:
                        LogFileReader.logValidation(
                            LogFileReader.Validation(cost=performanceMeasurer.cost, nodeAccuracy=performanceMeasurer.total_acc, nodeZeros=performanceMeasurer.total_zeros, rootAccuracy=performanceMeasurer.root_acc, rootZeros=performanceMeasurer.root_zeros),
                            LogFileReader.ValidationBest(cost=performanceMeasurerBest.cost, rootAccuracy=performanceMeasurerBest.root_acc, epoch=performanceMeasurerBest.epoch),
                            epoch=epoch)
                        cm = performanceMeasurer.total_confusion_matrix
                        if performanceMeasurer.total_nodes != cm[0] + cm[1] + cm[2] + cm[3]:
                            raise Exception("Expected total_node_count to be equal to sum", performanceMeasurer.total_nodes, cm[0] + cm[1] + cm[2] + cm[3])
                        print("Confusion Matrix total (tp,fp,tn,fn)", cm[0], cm[1], cm[2], cm[3])
                        cm = performanceMeasurer.root_confusion_matrix
                        if performanceMeasurer.total_root_nodes != cm[0] + cm[1] + cm[2] + cm[3]:
                            raise Exception("Expected total_root_node_count to be equal to sum", performanceMeasurer.total_root_nodes, cm[0] + cm[1] + cm[2] + cm[3])
                        print("Confusion Matrix root (tp,fp,tn,fn)", cm[0], cm[1], cm[2], cm[3])
                        # performanceMeasurerTrain = self.evaluate_model(train_trees, rnnWrapper, validate_model, cost_model)
                        # performanceMeasurerTrain.report(msg="{} Epoch {}. On train set: Current:".format(
                        #     datetime.now().strftime('%d%m%y %H:%M'), epoch))
                    if performanceMeasurerBest.root_acc < performanceMeasurer.root_acc:
                        filename = "{}_best.txt".format(file_prefix)
                        self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                        performanceMeasurerBest = performanceMeasurer
                        performanceMeasurerBest.running_epoch = epoch
                    else:
                        if performanceMeasurerBest.running_epoch + 1 < epoch:
                            filename = "{}_running.txt".format(file_prefix)
                            self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                            performanceMeasurerBest.running_epoch = epoch
                if output_running_model != -1 and it % output_running_model == 0:
                    filename = "{}_running_{}.txt".format(file_prefix, it)
                    self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
#                    while gc.collect() > 0:
#                        pass
        filename = "{}_running.txt".format(file_prefix)
        self.save(rnnWrapper=rnnWrapper, filename=filename, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)

    def save(self, rnnWrapper, filename, epoch, performanceMeasurer, performanceMeasurerBest):
        print("Saving rnnWrapper. Previous {};{:.4f}. New {};{:.4f}".format(performanceMeasurerBest.epoch, performanceMeasurerBest.root_acc, performanceMeasurer.epoch, performanceMeasurer.root_acc))
        print("Saving as " + filename)
        rnnWrapper.save(filename, epoch, performanceMeasurer.root_acc)

class RNNWrapper:
    def __init__(self, rng=RandomState(1234), cost_weight=0):
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.z = T.matrix('z', dtype=theano.config.floatX)    # for dropout
        # Define RNN
        self.rnn = nn_model.RNN(rng=rng, X=self.x, Z=self.z, n_in=2 * (rnn_enron.Evaluator.SIZE + rnn_enron.Evaluator.HIDDEN_SIZE),
                                n_hidden=rnn_enron.Evaluator.HIDDEN_SIZE, n_out=rnn_enron.Evaluator.RES_SIZE,
                                cost_weight=cost_weight)
        self.y_pred = self.rnn.y_pred

        self.run_model = theano.function(
            inputs=[self.x, self.z],
            outputs=self.y_pred)

    def load(self, filename='models/model.save'):
        return nn_model.load(rnn=self.rnn, filename=filename)

    def save(self, filename='models/model.save', epoch=0, acc=0):
        nn_model.save(rnn=self.rnn, filename=filename, epoch=epoch, acc=acc)

class IteratorGuard:
    def __init__(self):
        pass

def get_balanced_data(trees, rng, state=None):
    zero_trees = []
    four_trees = []
    count = 0
    try:
        for t in trees:
            if t.syntax == '0':
                zero_trees.append(t)
            elif t.syntax == '4':
                four_trees.append(t)
            count += 1
    except AttributeError:
        print("attribute error in loop len(trees) {}, count {}, type trees {}, type t {} type t {}".format(len(trees), count, type(trees), type(t), type(t)))
        print("type next(t) {}".format(type(next(t, IteratorGuard()))))
        raise
    if len(zero_trees) + len(four_trees) != len(trees):
        raise Exception("expected lengths to match {} + {} == {}".format(len(zero_trees), len(four_trees), len(trees)))
    min_list = zero_trees
    max_list = four_trees
    if (len(zero_trees) > len(four_trees)):
        min_list, max_list = four_trees, zero_trees
    if len(min_list) < 2:
        print("for training no examples in mimimum list")
        return []

    length = int(len(min_list) * 1)
    length = min(length, len(max_list))
    choices = rng.choice(len(max_list), size=length, replace=False)
    max_list = [max_list[i] for i in choices]
    if len(max_list) < 3:
        print("after pruning only: {} elements in list".format(2 * len(max_list)))
        return []
    res = min_list
    res.extend(max_list)
    # print("len(res)", len(res))
    perm = rng.permutation(len(res))
    res = [res[i] for i in perm]
    return res

# train = load_trees.get_trees('trees/train.txt')
# rnn_enron.initializeTrees(train, state.LT)
# train_trees = train[100:200]
# (list_root_indexes, x_val, y_val) = rnn_enron.getInputArrays(rnn.rnn, train_trees, evaluator)
def get_predictions(rnn, indexed_sentences):
    evaluator = rnn_enron.Evaluator(rnn.rnn)
    trees = []
    for s in indexed_sentences:
        trees.append(s.tree)
    (list_root_indexes, x_val, y_val) = rnn_enron.getInputArrays(rnn.rnn, trees, evaluator)
    x_roots = []
    y_roots = []
    for r in list_root_indexes:
        x_roots.append(x_val[r, :])
        y_roots.append(y_val[r, :])
    z_roots = numpy.ones(shape=(len(x_roots), rnn_enron.Evaluator.HIDDEN_SIZE))
    z_roots = z_roots * 0.9
    z_roots = z_roots.astype(dtype=theano.config.floatX)

    pred = rnn.run_model(x_roots, z_roots)
    for i in range(len(pred)):
        indexed_sentences[i].pred = pred[i]


if __name__ == "__main__":
    # testing
    import server_rnn_helper
    state = State()
    rnn = RNNWrapper()
    rnn.load('models/model.save')

    text = """Please have a look at enclosed worksheets.
    As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact. This may impact the final decision.
    We may have to move to cash margining if necessary."""
    indexed_sentences = server_rnn_helper.get_indexed_sentences(text)
    trees = server_rnn_helper.get_nltk_trees(0, indexed_sentences)
    rnn_enron.initializeTrees(trees, state.LT)
    get_predictions(rnn, indexed_sentences)

    for s in indexed_sentences:
        print(s.pred)
