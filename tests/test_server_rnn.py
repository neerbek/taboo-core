# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:26:30 2017

@author: neerbek
"""


import unittest

from numpy.random import RandomState

import similarity.load_trees as load_trees
import server_rnn

import tests.RunTimer

class TreeTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)
        
    def test_balanced_tree(self):
        trees = load_trees.get_trees(file = "tests/resources/trees_201_100_custom_0000_0250.txt", max_count=1500)
        print(type(trees))
        rng = RandomState(93)
        btrees = server_rnn.get_balanced_data(trees, rng)
        zero_trees = [t for t in btrees if t.syntax=='0']
        four_trees = [t for t in btrees if t.syntax=='4']
        self.assertEqual(25, len(zero_trees))
        self.assertEqual(25, len(four_trees))

        print("0", "4", len(zero_trees), len(four_trees))



if __name__ == "__main__":
    unittest.main()
