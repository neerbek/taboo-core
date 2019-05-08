# -*- coding: utf-8 -*-
"""
Created on August 16 2017

@author: neerbek
"""

import unittest
import os

from numpy.random import RandomState

import tests.RunTimer
import embedding
import confusion_matrix

import similarity.load_trees as load_trees

class ThresholdTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_get_cm(self):
        rng = RandomState(1234)
        s = 100
        dist = rng.uniform(0, 1, size=s * s)
        dist = dist.reshape(s, s)
        ground_truths = rng.randint(low=0, high=2, size=s)
        is_correct = rng.randint(low=0, high=2, size=s)
        sen_scores = rng.uniform(low=0, high=0.5, size=s)
        # print("is_correct", is_correct)
        # print("sen_scores", sen_scores)
        lines = []
        for i in range(s):
            l = confusion_matrix.Line(tree=load_trees.Node(), emb=None, ground_truth=ground_truths[i], is_correct=is_correct[i], sen_score=sen_scores[i])
            if l.ground_truth == 0:
                if l.is_correct == 1:
                    if l.sen_score > 0.5:
                        l.sen_score -= 0.5
                else:
                    if l.sen_score < 0.5:
                        l.sen_score += 0.5
            else:
                if l.is_correct == 1:
                    if l.sen_score < 0.5:
                        l.sen_score += 0.5
                else:
                    if l.sen_score > 0.5:
                        l.sen_score -= 0.5
            lines.append(l)
        # data generated, now test stuff
        indices = embedding.get_sort_indices_desc(dist)
        valid_cm = []
        for i in range(s):
            cm = confusion_matrix.finding_cm(dist, indices, i, lines, closest_count=40)
            valid_cm.append(cm)
        self.assertEqual(s, len(valid_cm))
        # for i in range(0, 100):
        #     cm = valid_cm[i]
        #     (x, pos, neg) = confusion_matrix.find_threshold(cm)
        #     print(i, cm.t_pos, cm.t_neg)
        i = 28
        cm = valid_cm[i]
        (x, pos, neg) = confusion_matrix.find_threshold(cm)
        self.assertEqual(0, cm.t_neg)
        self.assertEqual(51, cm.t_pos)
        # print(pos)
        y1 = [p[0] if p[0] < 100 else 100 for p in pos]
        y2 = [p[1] if p[1] < 100 else 100 for p in pos]
        y3 = [p[0] if p[0] < 100 else 100 for p in neg]
        y4 = [p[1] if p[1] < 100 else 100 for p in neg]
        if os.getenv('DISPLAY') != None and os.getenv('TABOO_CORE_NO_LATEX') == None:
            confusion_matrix.plot_graphs(x, y1, y2, y3, y4, "acc_random_test")


if __name__ == "__main__":
    unittest.main()
