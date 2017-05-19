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

    trees = None
    def get_trees(self):
        if RNNTest.trees == None:
           RNNTest.trees = load_trees.get_trees(file = "tests/resources/trees_201_100_custom_0000_0250.txt", max_count=2000)
        
        res = []
        for t in RNNTest.trees:
            res.append(load_trees.clone_tree(t))
        return res
        
    def test_no_empty_trees(self):
        all_trees = self.get_trees()
        self.assertEqual(2001, len(all_trees))
        for t in all_trees:
            self.assertNotEqual(0, load_trees.count_non_leaf_nodes(t), "expected count to be bigger than 0")
    
    def test_root_indexes(self):
        all_trees = self.get_trees()
        self.assertEqual(2001, len(all_trees))
        rng = RandomState(93574836)
        nx= 50
        unknown = rng.uniform(-1, 1, size=nx)
        LT = {}  #word embeddings
        LT[rnn_enron.UNKNOWN_WORD] = unknown

        rnn_enron.initializeTrees(all_trees, LT)
        rnn_enron.Evaluator.set_size(nx,nx)
        rnnWrapper = server_rnn.RNNWrapper(rng=rng)
                
        evaluator = rnn_enron.Evaluator(rnnWrapper.rnn)
        (list_root_indexes, x_val, y_val) = rnn_enron.getInputArrays(rnnWrapper.rnn, all_trees, evaluator)
        r = 0
        for i in range(len(all_trees)):
            t = all_trees[i]
            c = load_trees.count_non_leaf_nodes(t)
            r += c
            self.assertNotEqual(0, c, "expected node count to be bigger than 0")            
            self.assertEqual(r-1, list_root_indexes[i], "root count is unexpected")
                   
    def test_train(self):
        all_trees = self.get_trees()
        #all_trees = res
        self.assertEqual(2001, len(all_trees))
        f = load_trees.get_fraction(all_trees)
        self.assertAlmostEqual(0.23, f, places = 2)
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
        rng = RandomState(93)
        train_trees = rng.permutation(train_trees)
        dev_trees = rng.permutation(dev_trees)
        test_trees = rng.permutation(test_trees)
        
        train_trees=train_trees[200:300]
        #report ratio+size
        f = load_trees.get_fraction(train_trees)
        self.assertAlmostEqual(0.22, f, places = 2)
        f = load_trees.get_fraction(dev_trees)
        self.assertAlmostEqual(0.21, f, places = 2)
        f = load_trees.get_fraction(test_trees)
        self.assertAlmostEqual(0.22, f, places = 2)
        self.assertEqual(100, len(train_trees))
        self.assertEqual(400, len(dev_trees))
        self.assertEqual(401, len(test_trees))
        
        #trainer
        trainer = server_rnn.Trainer()
        trainer.learning_rate=5.5
        trainer.L1_reg=0.0
        trainer.L2_reg=0.0
        trainer.n_epochs=21
        trainer.batch_size=100
        trainer.retain_probability = 0.9
        
        #initialize state for training
        rng = RandomState(464)
        nx= 50
        nh = 100
        glove_path = "../code/glove"
        state = server_rnn.State(max_embedding_count=30000, nx = nx, nh = nh, rng = rng, glove_path=glove_path)
        state.train_trees = train_trees
        state.valid_trees = dev_trees
        state.test_trees = test_trees
        state.init_trees(trainer)
        
        rnnWrapper = server_rnn.RNNWrapper(rng = RandomState(958589))
        #load model here
        rng = RandomState(1234)
        # Training
        trainer.train(state = state, rnnWrapper = rnnWrapper, file_prefix="save_test", n_epochs = trainer.n_epochs, rng = rng, epoch=0)
        
        #measure 
        cost = trainer.get_cost(rnnWrapper=rnnWrapper)
        cost_model = trainer.get_cost_model(rnnWrapper=rnnWrapper, cost=cost)
        eval_model = trainer.get_validation_model(rnnWrapper=rnnWrapper)
        
        performanceMeasurer = trainer.evaluate_model(trees=dev_trees, rnnWrapper=rnnWrapper, validation_model = eval_model, cost_model = cost_model)
        m = max(performanceMeasurer.total_acc, performanceMeasurer.total_zeros, 1-performanceMeasurer.total_zeros)
        self.assertEquals(m, performanceMeasurer.total_acc, "expected total acc to be the strictly better than total_zeros/non_zeros")
        


if __name__ == "__main__":
    rnn_enron.DEBUG_PRINT=True
    server_rnn.DEBUG_PRINT=True
    unittest.main()
