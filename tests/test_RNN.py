# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:15:04 2017

@author: neerbek
"""

import unittest
from numpy.random import RandomState

import rnn_enron
import server_rnn
import ai_util
import similarity.load_trees as load_trees

import tests.RunTimer

class RNNTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_hello(self):
        self.assertEqual("42", "4"+"2", "I am very surprised that this test failed")

    def test_train(self):
        trainer = server_rnn.Trainer()
        trainer.learning_rate=0.01
        trainer.L1_reg=0.00
        trainer.L2_reg=0.0001
        trainer.n_epochs=1
        trainer.batch_size=400
        trainer.retain_probability = 0.4
        totaltimer = ai_util.Timer("Total time: ")
        traintimer =  ai_util.Timer("Train time: ")
        totaltimer.begin()
        all_trees = load_trees.get_trees(file = "tests/resources/trees_201_100_custom_0000_0250.txt", max_count=2000)
        load_trees.get_fraction(all_trees, report=True)
        #Constants to make the different set equal in ratios of sensitive vs non-sensitive
        test_index = int(0.15*len(all_trees))
        dev_index = int(0.28*len(all_trees))        
        dev_index2 = int(0.88*len(all_trees))        
        test_index2 = int(0.95*len(all_trees))
        
        test_trees = all_trees[:test_index]
        test_trees.extend(all_trees[test_index2:])
        dev_trees = all_trees[test_index:dev_index]
        dev_trees.extend(all_trees[dev_index2:test_index2])
        train_trees = all_trees[dev_index:dev_index2]
        
        
        #shuffel trees
        rng = RandomState(93574836)
        nx= 50
        nh = 100
        glove_path = "../code/glove"
        train_trees = rng.permutation(train_trees)
        dev_trees = rng.permutation(dev_trees)
        test_trees = rng.permutation(test_trees)
        
        train_trees=train_trees[:400]
        #report ratio+size
        load_trees.get_fraction(train_trees, report=True)
        load_trees.get_fraction(dev_trees, report=True)
        load_trees.get_fraction(test_trees, report=True)
        print(len(train_trees), len(dev_trees), len(test_trees))
        state = server_rnn.State(max_embedding_count=10000, nx = nx, nh = nh, rng = rng, glove_path=glove_path)
        state.train_trees = train_trees
        state.valid_trees = dev_trees
        state.test_trees = test_trees
        
        state.init_trees(trainer)
                
        rnnWrapper = server_rnn.RNNWrapper(rng = RandomState(34))
        #load model here
        rng = RandomState(1234)
        # Training
        traintimer.begin()
        trainer.train(state = state, rnnWrapper = rnnWrapper, file_prefix="save_test", n_epochs = trainer.n_epochs, rng = rng, epoch=0)
        traintimer.end()
        
        cost = trainer.get_cost(rnnWrapper=rnnWrapper)
        cost_model = trainer.get_cost_model(rnnWrapper=rnnWrapper, cost=cost)
        eval_model = trainer.get_validation_model(rnnWrapper=rnnWrapper)
        
        performanceMeasurer = trainer.evaluate_model(trees=train_trees, rnnWrapper=rnnWrapper, validation_model = eval_model, cost_model = cost_model)
        performanceMeasurer.report(msg="On train set")
        load_trees.get_fraction(train_trees, report=True)
        rnn_enron.get_zeros(train_trees)
        # Done
        totaltimer.end()
        totaltimer.report()
        traintimer.report()
        print("***train completed")

#epoch 1: 0.094936 (rng = RandomState(234))
#epoch 1. val total acc 59.7131 % (63.1169 %) val cost 0.094938, val root acc 77.2500 % (78.2500 %) (rng = RandomState(93574836))


if __name__ == "__main__":
    unittest.main()
