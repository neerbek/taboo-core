# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:42:23 2017

@author: neerbek
"""

import unittest

from numpy.random import RandomState

import numpy
import theano
import theano.tensor as T
# from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams

import rnn_model.rnn as nn_model
import rnn_model.learn as learn
import rnn_model.FlatTrainer

import tests.RunTimer


class RNNWrapper:
    def __init__(self):
        self.n_in = 10
        self.n_hidden = 8
        self.n_out = 5
        self.n_examples = 50
        self.retain_probability = 0.9
        self.rng = RandomState(1234)
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.z = T.matrix('z', dtype=theano.config.floatX)

    def create_rnn(self):
        self.rnn = nn_model.RNN(
            self.rng,
            self.x,
            self.z,
            n_in=self.n_in,
            n_hidden=self.n_hidden,
            n_out=self.n_out)
        c = self.rnn.cost(self.y)
        e = self.rnn.errors(self.y)
        self.validate_model = theano.function(
            inputs=[self.x, self.y, self.z], outputs=e)

        self.params = [
            self.rnn.reluLayer.W, self.rnn.reluLayer.b,
            self.rnn.regressionLayer.W, self.rnn.regressionLayer.b
        ]
        self.grads = [T.grad(cost=c, wrt=param) for param in self.params]

    def add_updates(self, updates):
        self.keys = [k for k in updates.keys()]

        self.train = theano.function(
            inputs=[self.x, self.y, self.z],
            outputs=[updates[k] for k in self.keys])

    def create_data(self):
        n_input_max_range = 10
        self.x_val = self.rng.randint(
            n_input_max_range, size=(self.n_examples, self.n_in))
        self.x_val = self.x_val.astype(dtype=theano.config.floatX)
        truth_val = self.rng.randint(self.n_out, size=self.n_examples)
        self.y_val = numpy.zeros(
            shape=(self.n_examples, self.n_out), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            self.y_val[i][truth_val[i]] = 1
        self.y_val = self.y_val.astype(dtype=theano.config.floatX)

        self.z_val = self.rng.binomial(
            n=1,
            size=(self.x_val.shape[0], self.n_hidden),
            p=self.retain_probability)
        self.z_val = self.z_val.astype(dtype=theano.config.floatX)

    def do_train(self, x_val, y_val, z_val):
        values = self.train(x_val, y_val, z_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])

class RNNWrapper2:
    def __init__(self):
        self.model = None
        self.n_examples = 50
        self.n_hidden = 8
        self.retain_probability = 0.9
        self.trainParam = None
        self.modelEvaluator = None
        self.keys = None
        self.train = None

    def create_rnn(self):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, isDropoutEnabled=True, rng=RandomState(1234))
        self.model.addLayer(rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=5))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)

    def add_updates(self, updates):
        self.keys = [k for k in updates.keys()]

        self.train = theano.function(
            inputs=[self.model.x, self.model.y],
            outputs=[updates[k] for k in self.keys])

    def create_data(self):
        n_input_max_range = 10
        self.x_val = self.model.rng.randint(
            n_input_max_range, size=(self.n_examples, self.model.nIn))
        self.x_val = self.x_val.astype(dtype=theano.config.floatX)
        truth_val = self.model.rng.randint(self.model.nOut, size=self.n_examples)
        self.y_val = numpy.zeros(
            shape=(self.n_examples, self.model.nOut), dtype=theano.config.floatX)
        for i in range(len(truth_val)):
            self.y_val[i][truth_val[i]] = 1
        self.y_val = self.y_val.astype(dtype=theano.config.floatX)

        self.z_val = self.model.rng.binomial(
            n=1,
            size=(self.x_val.shape[0], self.n_hidden),
            p=self.retain_probability)
        self.z_val = self.z_val.astype(dtype=theano.config.floatX)

    def do_train(self, x_val, y_val, z_val=None):
        values = self.train(x_val, y_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])

class RNNWrapper2_withdropout(RNNWrapper2):
    def __init__(self):
        RNNWrapper2.__init__(self)

    def create_rnn(self):
        self.model = RNNContainer3(nIn=10, isDropoutEnabled=True, rng=RandomState(1234))
        dropout = DropoutLayerOriginal(self.model, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=5))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam, inputs=[self.model.x, self.model.y, self.model.z])

    def add_updates(self, updates):
        self.keys = [k for k in updates.keys()]

        self.train = theano.function(
            inputs=[self.model.x, self.model.y, self.model.z],
            outputs=[updates[k] for k in self.keys])

    def do_train(self, x_val, y_val, z_val=None):
        values = self.train(x_val, y_val, z_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])

class RNNWrapper2_withrunningdropout(RNNWrapper2_withdropout):
    def __init__(self):
        RNNWrapper2_withdropout.__init__(self)

    def get_next_dropout(self):
        self.z_val = self.model.rng.binomial(
            n=1,
            size=(self.x_val.shape[0], self.n_hidden),
            p=self.retain_probability)
        self.z_val = self.z_val.astype(dtype=theano.config.floatX)
        return self.z_val

class DropoutLayerOriginal:
    def __init__(self, container, innerLayer):
        self.container = container
        self.z = container.z
        self.nOut = innerLayer.nOut
        self.params = []
        self.regularizedParams = []
        self.rng = None
        self.nIn = None
        self.innerLayer = innerLayer

    def clone(self, container):
        innerLayerClone = self.innerLayer.clone(container)
        return DropoutLayerOriginal(container, innerLayerClone)

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.rng = rng
        self.nIn = nIn
        self.innerLayer.setInputSize(nIn, x, layerNumber, rng)
        self.params = self.innerLayer.params
        self.regularizedParams = self.innerLayer.regularizedParams

    def getPrediction(self):
        return self.z * self.innerLayer.getPrediction()


class DropoutLayerV2:
    def __init__(self, container, retain_probability, innerLayer):
        self.container = container
        self.z = container.z
        self.retain_probability = retain_probability
        self.nOut = innerLayer.nOut
        self.params = []
        self.regularizedParams = []
        self.rng = None
        self.nIn = None
        self.innerLayer = innerLayer
        self.isTraining = True

    def clone(self, container):
        innerLayerClone = self.innerLayer.clone(container)
        return DropoutLayerV2(container, self.retain_probability, innerLayerClone)

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.rng = RandomStreams(seed=rng.randint(1000000))  # RandomStreams(seed=234)
        self.rngDropout = self.rng.binomial(n=1,
                                            size=(T.cast(self.z, dtype='int32'), self.nOut),
                                            p=self.retain_probability)
        self.rngDropout = T.cast(self.rngDropout, dtype='float32')

        self.nIn = nIn
        self.innerLayer.setInputSize(nIn, x, layerNumber, rng)
        self.params = self.innerLayer.params
        self.regularizedParams = self.innerLayer.regularizedParams

    def setIsTraining(self, isTraining=True):
        self.isTraining = isTraining

    def getPrediction(self):
        pred = self.innerLayer.getPrediction()
        dropout = 0
        if self.container.isDropoutEnabled:
            dropout = self.rngDropout
        else:
            scale = T.ones(shape=(T.cast(self.z, dtype='int32'), self.nOut))
            scale = self.retain_probability * scale
            dropout = scale
        return dropout * pred


class RNNContainer2(rnn_model.FlatTrainer.RNNContainer):
    def __init__(self, nIn, isDropoutEnabled, rng=RandomState(1234)):
        rnn_model.FlatTrainer.RNNContainer.__init__(self, nIn, isDropoutEnabled, rng)
        self.z = T.scalar('z', dtype=theano.config.floatX)    # for dropout

    def clone(self, isDropoutEnabled, rng=RandomState(1234)):
        clone = RNNContainer2(self.nIn, isDropoutEnabled, rng)
        # copy of loop from RNNContainer
        for l in self.layers:
            lClone = l.clone(clone)
            clone.addLayer(lClone)
            params = l.params
            cloneParams = lClone.params
            for i in range(len(params)):
                cloneParams[i].set_value(params[i].get_value())
        return clone

class RNNContainer3(rnn_model.FlatTrainer.RNNContainer):
    def __init__(self, nIn, isDropoutEnabled, rng=RandomState(1234)):
        rnn_model.FlatTrainer.RNNContainer.__init__(self, nIn, isDropoutEnabled, rng)
        self.z = T.matrix('z', dtype=theano.config.floatX)    # for dropout

    def clone(self, isDropoutEnabled, rng=RandomState(1234)):
        clone = RNNContainer2(self.nIn, isDropoutEnabled, rng)
        # copy of loop from RNNContainer
        for l in self.layers:
            lClone = l.clone(clone)
            clone.addLayer(lClone)
            params = l.params
            cloneParams = lClone.params
            for i in range(len(params)):
                cloneParams[i].set_value(params[i].get_value())
        return clone


class RNNWrapper2_withrunningdropout2(RNNWrapper2_withdropout):
    def __init__(self, retain_probability):
        RNNWrapper2_withdropout.__init__(self)
        self.retain_probability = retain_probability

    def create_rnn(self):
        self.model = RNNContainer2(nIn=10, isDropoutEnabled=True, rng=RandomState(1234))

        dropout = DropoutLayerV2(self.model, self.retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=5))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam, inputs=[self.model.x, self.model.y, self.model.z])

    def do_train(self, x_val, y_val, z_val=None):
        if z_val is None:
            z_val = x_val.shape[0]
        values = self.train(x_val, y_val, z_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])

    def get_next_dropout(self):
        return self.x_val.shape[0]

    def getAccuracy(self, x_val, y_val):
        return self.modelEvaluator.accuracyFunction(x_val, y_val, x_val.shape[0])


class RNNWrapper2_withrunningdropout3(RNNWrapper2_withdropout):
    def __init__(self, retain_probability):
        RNNWrapper2_withdropout.__init__(self)
        self.retain_probability = retain_probability

    def create_rnn(self):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, isDropoutEnabled=True, rng=RandomState(1234))
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model, self.retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=5))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam)

    def add_updates(self, updates):
        self.keys = [k for k in updates.keys()]

        self.train = theano.function(
            inputs=[self.model.x, self.model.y],
            outputs=[updates[k] for k in self.keys])

    def do_train(self, x_val, y_val, z_val=None):
        values = self.train(x_val, y_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])

    def getAccuracy(self, x_val, y_val):
        return self.modelEvaluator.accuracyFunction(x_val, y_val)


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_gd(self):
        lr = 0.4
        n_loops = 5000
        rnnWrapper = RNNWrapper()
        rnnWrapper.create_rnn()

        updates = learn.gd(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 1000 == 0:
            #     print("gd error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.1, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected gd error ratio")

    def test_gd_nodropout(self):
        lr = 0.4
        n_loops = 5000
        rnnWrapper = RNNWrapper()
        rnnWrapper.retain_probability = 1
        rnnWrapper.create_rnn()

        updates = learn.gd(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 1000 == 0:
            #     print("gd error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.04, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected gd error ratio")

    def test_gd2(self):
        n_loops = 5000
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentLearner(lr=0.4)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        # print(type(y_val))

        for i in range(n_loops):
            # if i % 1000 == 0:
            #     print("gd error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(0.04, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val), 2), "Mismatch in final expected gd2 error ratio")

    def test_gd2_withdropout(self):
        n_loops = 5000
        rnnWrapper = RNNWrapper2_withdropout()
        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentLearner(lr=0.4)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 1000 == 0:
            #     print("gd error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.1000, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val, z_val), 4), "Mismatch in final expected gd2_withdropout error ratio")

    def test_adaGrad(self):
        lr = 0.025
        n_loops = 2200
        rnnWrapper = RNNWrapper()
        rnnWrapper.create_rnn()

        updates = learn.adagrad(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 400 == 0:
            #     print("adagrad error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.04, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected adagrad error ratio")

    def test_adaGrad_nodropout(self):
        lr = 0.025
        n_loops = 2200
        rnnWrapper = RNNWrapper()
        rnnWrapper.retain_probability = 1
        rnnWrapper.create_rnn()

        updates = learn.adagrad(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 400 == 0:
            #     print("adagrad error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.1, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected adagrad error ratio")

    def test_adaGrad_nodropout2(self):
        lr = 0.025
        n_loops = 2200
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.AdagradLearner(lr)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        for i in range(n_loops):
            # if i % 400 == 0:
            #     print("adagrad error ratio",
            #           rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(0.10, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val), 3), "Mismatch in final expected adagrad error ratio")

    def test_adaGrad_withdropout2(self):
        lr = 0.025
        n_loops = 2200
        rnnWrapper = RNNWrapper2_withdropout()
        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.AdagradLearner(lr)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 400 == 0:
            #     print("adagrad error ratio",
            #           rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.0400, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val, z_val), 4), "Mismatch in final expected adagrad error ratio")

    @unittest.skip
    def test_adaGrad_nodropout_large(self):
        lr = 0.025
        n_loops = 5000
        rnnWrapper = RNNWrapper()
        rnnWrapper.retain_probability = 1
        rnnWrapper.n_examples = 5000
        rnnWrapper.n_hidden = 200
        rnnWrapper.create_rnn()

        updates = learn.adagrad(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            if i % 400 == 0:
                print("adagrad error ratio",
                      rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.4678, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected adagrad error ratio")

    @unittest.skip
    def test_adaGrad_nodropout_large2(self):
        lr = 0.025
        n_loops = 5000
        rnnWrapper = RNNWrapper2()
        rnnWrapper.n_examples = 5000
        rnnWrapper.n_hidden = 200
        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.AdagradLearner(lr)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        for i in range(n_loops):
            if i % 400 == 0:
                print("adagrad error ratio",
                      rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(0.4678, 1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val), "Mismatch in final expected adagrad error ratio")

    def test_gd_momentum(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper()
        rnnWrapper.create_rnn()

        updates = learn.gd_momentum(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr, mc=mc)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.04, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected error")

    def test_gd_momentum_nodropout(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper()
        rnnWrapper.retain_probability = 1
        rnnWrapper.create_rnn()

        updates = learn.gd_momentum(
            params=rnnWrapper.params, grads=rnnWrapper.grads, lr=lr, mc=mc)
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.08, rnnWrapper.validate_model(x_val, y_val, z_val), "Mismatch in final expected error")

    def test_gd_momentum_nodropout2(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper2()
        rnnWrapper.create_rnn()

        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(0.08, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val), 2), "Mismatch in final expected gdm error")

    def test_gd_momentum_withdropout2(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper2_withdropout()
        rnnWrapper.create_rnn()

        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val
        z_val = rnnWrapper.z_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.0400, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val, z_val), 4), "Mismatch in final expected gdm error")

    def test_gd_momentum_withrunningdropout2(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper2_withrunningdropout()
        rnnWrapper.create_rnn()

        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        z_val = None
        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            z_val = rnnWrapper.get_next_dropout()
            rnnWrapper.do_train(x_val, y_val, z_val)
        z_val = numpy.ones(z_val.shape)
        z_val = rnnWrapper.retain_probability * z_val
        z_val = z_val.astype(dtype=theano.config.floatX)
        self.assertEqual(0.1200, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val, z_val), 4), "Mismatch in final expected gdm error")

    def test_gd_momentum_withrunningdropout2_2(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper2_withrunningdropout2(retain_probability=0.8)
        rnnWrapper.create_rnn()

        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        z_val = rnnWrapper.get_next_dropout()
        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val, z_val)
        self.assertEqual(0.2800, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val, z_val), 4), "Mismatch in final expected gdm error")

    def test_gd_momentum_withrunningdropout2_3(self):
        lr = 0.75
        mc = 0.0009
        n_loops = 3000
        rnnWrapper = RNNWrapper2_withrunningdropout3(retain_probability=0.8)
        rnnWrapper.create_rnn()

        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(0.2800, numpy.around(1 - rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val), 4), "Mismatch in final expected gdm error")

    def gd_momentum_run_validator(self, rnnWrapper, lr, mc, res):
        n_loops = 3000

        rnnWrapper.create_rnn()
        rnnWrapper.trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr, mc)
        updates = rnnWrapper.trainParam.learner.getUpdates(rnnWrapper.model.getParams(), rnnWrapper.modelEvaluator.cost())
        rnnWrapper.add_updates(updates)
        rnnWrapper.create_data()
        x_val = rnnWrapper.x_val
        y_val = rnnWrapper.y_val

        for i in range(n_loops):
            # if i % 600 == 0:
            #     print("gd_m error ratio",
            #           rnnWrapper.validate_model(x_val, y_val, z_val))
            rnnWrapper.do_train(x_val, y_val)
        self.assertEqual(res, numpy.around(1 - rnnWrapper.getAccuracy(x_val, y_val), 4), "Mismatch in final expected gdm error")

    def test_momentumWrapper(self):
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout3(retain_probability=0.8), 0.75, 0.0009, 0.2800)
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout2(retain_probability=0.8), 0.75, 0.0009, 0.2800)
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout3(retain_probability=0.9), 0.75, 0.0009, 0.1800)
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout2(retain_probability=0.9), 0.75, 0.0009, 0.1800)
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout3(retain_probability=1.0), 0.75, 0.0009, 0.1200)
        self.gd_momentum_run_validator(RNNWrapper2_withrunningdropout2(retain_probability=1.0), 0.75, 0.0009, 0.1200)

    def test_momentumWrapper2(self):
        rnnWrapper2 = RNNWrapper2_withrunningdropout2(retain_probability=0.8)
        self.gd_momentum_run_validator(rnnWrapper2, 0.75, 0.0009, 0.2800)
        rnnWrapper3 = RNNWrapper2_withrunningdropout3(retain_probability=0.8)
        self.gd_momentum_run_validator(rnnWrapper3, 0.75, 0.0009, 0.2800)
        self.assertEqual(0.2600, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        self.assertEqual(0.2600, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2800, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        self.assertEqual(0.2800, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2600, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        self.assertEqual(0.2600, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        rnnWrapper2Clone = RNNWrapper2_withrunningdropout2(retain_probability=0.8)
        rnnWrapper2Clone.model = rnnWrapper2.model.clone(isDropoutEnabled=False)
        rnnWrapper2Clone.trainParam = rnnWrapper2.trainParam
        rnnWrapper2Clone.x_val = rnnWrapper2.x_val
        rnnWrapper2Clone.y_val = rnnWrapper2.y_val
        rnnWrapper2 = rnnWrapper2Clone
        rnnWrapper2.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper2.model, rnnWrapper2.trainParam, inputs=[rnnWrapper2.model.x, rnnWrapper2.model.y, rnnWrapper2.model.z])
        rnnWrapper3Clone = RNNWrapper2_withrunningdropout3(retain_probability=0.8)
        rnnWrapper3Clone.model = rnnWrapper3.model.clone(isDropoutEnabled=False)
        rnnWrapper3Clone.trainParam = rnnWrapper3.trainParam
        rnnWrapper3Clone.x_val = rnnWrapper3.x_val
        rnnWrapper3Clone.y_val = rnnWrapper3.y_val
        rnnWrapper3 = rnnWrapper3Clone
        rnnWrapper3.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper3.model, rnnWrapper3.trainParam)
        # print(pp(rnnWrapper2.modelEvaluator.accuracy()))
        # rnnWrapper3.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(rnnWrapper3.model, rnnWrapper3.trainParam, withDropout=False)

        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2200, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2800, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2200, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2200, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2200, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))
        self.assertEqual(0.2200, numpy.around(1 - rnnWrapper2.getAccuracy(rnnWrapper2.x_val, rnnWrapper2.y_val), 4))
        # self.assertEqual(0.2200, numpy.around(1 - rnnWrapper3.getAccuracy(rnnWrapper3.x_val, rnnWrapper3.y_val), 4))


if __name__ == "__main__":
    unittest.main()
