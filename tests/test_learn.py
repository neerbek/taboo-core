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

    def create_rnn(self):
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, rng=RandomState(1234))
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
        self.model = rnn_model.FlatTrainer.RNNContainer(nIn=10, rng=RandomState(1234))
        dropout = rnn_model.FlatTrainer.DropoutLayer(self.model.z, rnn_model.FlatTrainer.ReluLayer(nOut=self.n_hidden))
        self.model.addLayer(dropout)
        self.model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=5))
        self.trainParam = rnn_model.FlatTrainer.TrainParam()
        self.modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(self.model, self.trainParam, withDropout=True)

    def add_updates(self, updates):
        self.keys = [k for k in updates.keys()]

        self.train = theano.function(
            inputs=[self.model.x, self.model.y, self.model.z],
            outputs=[updates[k] for k in self.keys])

    def do_train(self, x_val, y_val, z_val=None):
        values = self.train(x_val, y_val, z_val)
        for index, param in enumerate(self.keys):
            param.set_value(values[index])


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
        print(type(y_val))

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
            if i % 400 == 0:
                print("adagrad error ratio",
                      rnnWrapper.modelEvaluator.accuracyFunction(x_val, y_val))
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


if __name__ == "__main__":
    unittest.main()
