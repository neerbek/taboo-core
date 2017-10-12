# -*- coding: utf-8 -*-
"""

Created on October 9, 2017

@author:  neerbek
"""

import unittest

import numpy
from numpy.random import RandomState
import theano

import tests.RunTimer

import rnn_model.learn
import rnn_model.FlatTrainer

class RNNWrapper:
    def __init__(self):
        self.model = None
        self.n_examples = 50
        self.n_hidden = 8
        self.retain_probability = 0.9
        self.trainParam = None
        self.modelEvaluator = None

    def create_rnn(self):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, isDropoutEnabled=True, rng=RandomState(1234))
        self.model.addLayer(rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=2))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)

    def create_data(self):
        n_input_max_range = 10
        self.trainParam.X = self.model.rng.randint(
            n_input_max_range, size=(self.n_examples, self.model.nIn))
        self.trainParam.X = self.trainParam.X.astype(dtype=theano.config.floatX)
        truth_val = self.model.rng.randint(self.model.nOut, size=self.n_examples)
        self.trainParam.Y = numpy.zeros(
            shape=(self.n_examples, self.model.nOut), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            self.trainParam.Y[i][truth_val[i]] = 1
        self.trainParam.Y = self.trainParam.Y.astype(dtype=theano.config.floatX)
        self.trainParam.valX = self.model.rng.randint(
            n_input_max_range, size=(self.n_examples, self.model.nIn))
        self.trainParam.valX = self.trainParam.valX.astype(dtype=theano.config.floatX)
        truth_val = self.model.rng.randint(self.model.nOut, size=self.n_examples)
        self.trainParam.valY = numpy.zeros(
            shape=(self.n_examples, self.model.nOut), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            self.trainParam.valY[i][truth_val[i]] = 1
        self.trainParam.valY = self.trainParam.valY.astype(dtype=theano.config.floatX)
        self.trainParam.batchSize = 10
        self.trainParam.learner = rnn_model.learn.GradientDecentLearner(lr=0.4)

class RNNWrapper2(RNNWrapper):
    def __init__(self):
        RNNWrapper.__init__(self)
        self.trainParam = rnn_model.FlatTrainer.TrainParam()

    def create_rnn(self, isDropoutEnabled=True):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, isDropoutEnabled=isDropoutEnabled, rng=RandomState(1234))
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model, self.retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=2))
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)

    def create_multilayer_rnn(self, isDropoutEnabled=True):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, isDropoutEnabled=isDropoutEnabled, rng=RandomState(1234))
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model, 0.5, rnn_model.FlatTrainer.ReluLayer(nOut=2 * self.n_hidden))
        self.model.addLayer(dropout)
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model, 0.6, rnn_model.FlatTrainer.ReluLayer(nOut=int(1.5 * self.n_hidden)))
        self.model.addLayer(dropout)
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model, self.retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=2))
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)

    def cloneModel(self, isDropoutEnabled):
        originalModel = self.model
        if len(self.model.layers) == 2:
            self.create_rnn(isDropoutEnabled)
        else:
            self.create_multilayer_rnn(isDropoutEnabled)
        clone = self.model
        self.model = originalModel
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)
        self.model.updateClone(clone)
        return clone

class TrainTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def getRandFunction(rnnWrapper):
        return theano.function(
            inputs=[rnnWrapper.model.x],
            outputs=rnnWrapper.model.layers[0].getNewRandom()
        )

    def test_train(self):
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()
        randFunction = TrainTest.getRandFunction(rnnWrapper)

        rnnWrapper.create_data()
        rnnWrapper.trainParam.batchSize = 50
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer=valContainer, n_epochs=2, trainReportFrequency=1, validationFrequency=1)
        print(type(rnnWrapper), type(rnnWrapper.model), type(rnnWrapper.modelEvaluator))
        print("rand1", randFunction(rnnWrapper.trainParam.valX)[0:4, :])
        rnnWrapper.model = rnnWrapper.cloneModel(True)
        rnnWrapper.model.load(filename="save_running.txt")
        valContainer = rnnWrapper.cloneModel(False)
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.4400, numpy.around(acc, 4))
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=5, validationFrequency=1, epoch=3)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4200, numpy.around(acc, 4))  # training, expect acc to change
        self.assertEqual(0.4800, numpy.around(acc, 4))  # training, expect acc to change
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=7, validationFrequency=3, trainReportFrequency=2, epoch=6)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.3800, numpy.around(acc, 4))   # training, expect acc to change
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.3800, numpy.around(acc, 4))   # because modelEvaluator points to old theano graph
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4000, numpy.around(acc, 4))  # updated model
        self.assertEqual(0.3800, numpy.around(acc, 4))  # updated model
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4000, numpy.around(acc, 4))  # deterministic eval
        self.assertEqual(0.3800, numpy.around(acc, 4))  # deterministic eval
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4000, numpy.around(acc, 4))  # deterministic eval
        self.assertEqual(0.3800, numpy.around(acc, 4))  # deterministic eval
        rnnWrapper.model = rnnWrapper.cloneModel(True)
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4400, numpy.around(acc, 4))  # dropout enabled, random results
        self.assertEqual(0.5000, numpy.around(acc, 4))  # dropout enabled, random results
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4200, numpy.around(acc, 4))  # dropout enabled
        self.assertEqual(0.4400, numpy.around(acc, 4))  # dropout enabled
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.4400, numpy.around(acc, 4))  # dropout enabled, random results
        self.assertEqual(0.3600, numpy.around(acc, 4))  # dropout enabled, random results
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        print("test done")

    def test_validation_measurement(self):
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()
        rnnWrapper.create_data()
        rnnWrapper.trainParam.batchSize = 50
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=6, validationFrequency=3, trainReportFrequency=3)
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        rnnWrapper.model.load(filename="save_running.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.4400, numpy.around(acc, 4))
        # self.assertEqual(0.4600, numpy.around(acc, 4))
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        rnnWrapper.model.load(filename="save_best.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.4400, numpy.around(acc, 4))
        # self.assertEqual(0.4600, numpy.around(acc, 4))  # should be unchanged (no dropout)
        rnnWrapper.model = rnnWrapper.cloneModel(True)
        rnnWrapper.model.load(filename="save_best.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=100, validationFrequency=50, trainReportFrequency=25)
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        rnnWrapper.model.load(filename="save_running.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        # self.assertEqual(0.5200, numpy.around(acc, 4))
        self.assertEqual(0.5400, numpy.around(acc, 4))
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        rnnWrapper.model.load(filename="save_best.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(0.5400, numpy.around(acc, 4))

    def test_train_multi(self):
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_multilayer_rnn()
        rnnWrapper.create_data()
        rnnWrapper.trainParam.batchSize = 50
        nEpochs = 4000
        # (expCost, expAcc, lr, mc) = (0.1244, 0.5600, 1.5, 0.2)
        # (expCost, expAcc, lr, mc) = (0.1247, 0.5600, 1.5, 0.6)
        # (expCost, expAcc, lr, mc) = (0.1246, 0.5600, 1.5, 0.4)
        # (expCost, expAcc, lr, mc) = (0.1243, 0.5600, 1.5, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1243, 0.5600, 1.25, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1243, 0.5600, 1.1, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1865, 0.5200, 1.1, 0)
        # (expCost, expAcc, lr, mc) = (0.1865, 0.5000, 1.5, 0)
        # (expCost, expAcc, lr, mc) = (0.1865, 0.4800, 2.1, 0)
        # (expCost, expAcc, lr, mc) = (0.1239, 0.5600, 15, 0.1)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1243, 0.5600, 1.1, 0.1, 8000)
        (expCost, expAcc, lr, mc, nEpochs) = (0.1243, 0.5600, 1.1, 0.1, 16000)  # t = 9.1; 9.1
        # (expCost, expAcc, lr, mc, nEpochs) = (numpy.nan, 0.5600, 1.1, 1.1, 8000)   # nan
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1247, 0.5600, 1.1, 0.7, 1000)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1247, 0.5600, 1.1, 0.7, 800)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1247, 0.5600, 1.1, 0.7, 400)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1247, 0.5600, 1.1, 0.7, 100)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1250, 0.5600, 1.1, 0.7, 25)
        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr=lr, mc=mc)
        # rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentLearner(lr=lr)
        rng = RandomState(1234)
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=nEpochs, validationFrequency=4000, trainReportFrequency=4000, rng=rng)
        # rnnWrapper.model = rnnWrapper.model.clone(False)
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        rnnWrapper.model.load(filename="save_running.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        print("test training completed,", lr, mc)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(expAcc, numpy.around(acc, 4))
        cost = rnnWrapper.modelEvaluator.costFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        self.assertEqual(int(expCost * 10000), int(numpy.around(cost, 4) * 10000))

    def test_train_single(self):
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()
        rnnWrapper.create_data()
        rnnWrapper.trainParam.batchSize = 50
        nEpochs = 4000
        # (expCost, expAcc, lr, mc) = (0.2203, 0.4800, 1.1, 0)
        # (expCost, expAcc, lr, mc) = (0.2255, 0.4800, 1.5, 0)
        # (expCost, expAcc, lr, mc) = (0.2066, 0.4800, 2.1, 0)
        # (expCost, expAcc, lr, mc) = (0.1271, 0.4800, 1.5, 0.2)
        # (expCost, expAcc, lr, mc) = (0.1256, 0.4800, 1.5, 0.6)
        # (expCost, expAcc, lr, mc) = (0.1258, 0.4800, 1.5, 0.4)
        # (expCost, expAcc, lr, mc) = (0.1328, 0.4800, 1.5, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1293, 0.4800, 1.25, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1295, 0.4600, 1.1, 0.1)
        # (expCost, expAcc, lr, mc) = (0.1294, 0.4800, 15, 0.1)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1286, 0.4800, 1.1, 0.1, 8000)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1282, 0.4800, 1.1, 0.1, 16000)
        # (expCost, expAcc, lr, mc, nEpochs) = (nan, 0.5200, 1.1, 1.1, 8000)   # nan
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1253, 0.4800, 1.1, 0.7, 1000)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1253, 0.4800, 1.1, 0.7, 800)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1253, 0.4800, 1.1, 0.7, 400)
        # (expCost, expAcc, lr, mc, nEpochs) = (0.1253, 0.4800, 1.1, 0.7, 100)
        (expCost, expAcc, lr, mc, nEpochs) = (0.1254, 0.4800, 1.1, 0.7, 25)
        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr=lr, mc=mc)
        # rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentLearner(lr=lr)
        rng = RandomState(1234)
        valContainer = rnnWrapper.cloneModel(False)
        rnn_model.FlatTrainer.train(rnnWrapper.trainParam, rnnWrapper.model, valContainer, n_epochs=nEpochs, validationFrequency=1000, trainReportFrequency=1000, rng=rng)
        rnnWrapper.model = rnnWrapper.cloneModel(False)
        # rnnWrapper.model = rnnWrapper.model.clone(False)
        rnnWrapper.model.load(filename="save_running.txt")
        rnnWrapper.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper.model, rnnWrapper.trainParam)
        acc = rnnWrapper.modelEvaluator.accuracyFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        cost = rnnWrapper.modelEvaluator.costFunction(rnnWrapper.trainParam.valX, rnnWrapper.trainParam.valY)
        print("test training completed,", lr, mc, numpy.around(acc, 4), numpy.around(cost, 4))
        self.assertEqual(expAcc, numpy.around(acc, 4))
        self.assertEqual(int(expCost * 10000), int(numpy.around(cost, 4) * 10000))

    def verify_save_load(self, rnnWrapper):
        rnnWrapper.create_rnn()
        val = rnnWrapper.model.layers[0].params[0].get_value()
        val[9][7] = 3.141593
        rnnWrapper.model.layers[0].params[0].set_value(val)
        val = rnnWrapper.model.layers[1].params[0].get_value()
        val[2][0] = 3.141593
        rnnWrapper.model.layers[1].params[0].set_value(val)
        val = rnnWrapper.model.layers[0].params[1].get_value()
        val[7] = -3.141593
        rnnWrapper.model.layers[0].params[1].set_value(val)
        rnnWrapper.model.save(filename="save_running.txt", epoch=42, acc=0.98)
        rnnWrapper.model = None
        rnnWrapper.create_rnn()
        rnnWrapper.model.load(filename="save_running.txt")
        val = rnnWrapper.model.layers[0].params[0].get_value()
        self.assertEqual(31416, numpy.around(val[9][7] * 10000))
        val = rnnWrapper.model.layers[1].params[0].get_value()
        self.assertEqual(31416, numpy.around(val[2][0] * 10000))
        val = rnnWrapper.model.layers[0].params[1].get_value()
        self.assertEqual(-31416, numpy.around(val[7] * 10000))

    def test_save_load1(self):
        rnnWrapper = RNNWrapper()
        self.verify_save_load(rnnWrapper)

    def test_save_load2(self):
        rnnWrapper = RNNWrapper2()
        self.verify_save_load(rnnWrapper)
