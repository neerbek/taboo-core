# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:06:48 2017

@author: neerbek
"""

import unittest

import similarity.load_trees as load_trees

import tests.RunTimer


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_constructing(self):
        n = load_trees.Node()
        n2 = n.add_child()
        n2.word = "Hello"
        n3 = n.add_child()
        n3.word = "World"
        s = load_trees.output_sentence(n)
        self.assertEqual(" Hello World", s, "Tree sentence does not match")

    def test_count(self):
        n = load_trees.Node()
        n2 = n.add_child()
        n2.word = "Hello"
        n3 = n.add_child()
        n3.word = "World"
        c = load_trees.count_non_leaf_nodes(n)
        self.assertEqual(1, c, "count was wrong")

    def test_count2(self):
        n = load_trees.Node()
        n2 = n.add_child()
        n2.add_child()
        n22 = n2.add_child()
        n22.add_child()
        n22.add_child()
        n.add_child()
        c = load_trees.count_non_leaf_nodes(n)
        self.assertEqual(3, c, "count was wrong")

    def test_count3(self):
        n = load_trees.Node()
        n2 = n.add_child()
        n21 = n2.add_child()
        n21.add_child()
        n22 = n2.add_child()
        n22.add_child()
        n22.add_child()
        n.add_child()
        c = load_trees.count_non_leaf_nodes(n)
        self.assertEqual(4, c, "count was wrong")

    def test_clone(self):
        n = load_trees.Node()
        n.syntax = "2"
        n2 = n.add_child()
        n2.syntax = "n2"
        n21 = n2.add_child()
        n21.syntax = "n21"
        n211 = n21.add_child()
        n211.syntax = "n211"
        n22 = n2.add_child()
        n22.syntax = "n22"
        n221 = n22.add_child()
        n221.syntax = "n221"
        n222 = n22.add_child()
        n222.syntax = "n222"
        n1 = n.add_child()
        n1.syntax = "n1"

        n_clone = load_trees.clone_tree(n)
        a1 = load_trees.as_array(n)
        a2 = load_trees.as_array(n_clone)
        for i in range(len(a1)):
            self.assertEqual(a1[i].syntax, a2[i].syntax,
                             "expected nodes to share values")


if __name__ == "__main__":
    unittest.main()
