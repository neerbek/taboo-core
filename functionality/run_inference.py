# -*- coding: utf-8 -*-
"""
Created on Jun 28 2017

@author: neerbek
"""

import sys
from numpy.random import RandomState


import ai_util
import rnn_enron
import inference_enron
import similarity.load_trees as load_trees


counttrees = None
inputtrees = None
cutoffs = [0]
max_tree_count = -1
random_seed = 1234

traintimer = ai_util.Timer("Model time: ")
totaltimer = ai_util.Timer("Total time: ")
totaltimer.begin()


def syntax():
    print(
        """syntax: run_inference.py [-counttrees <trees>][-inputtrees <trees>]
    [-cutoff <float>][-max_tree_count <int>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2
    if setting == '-inputtrees':
        inputtrees = arg
    elif setting == '-counttrees':
        counttrees = arg
    elif setting == '-cutoff':
        cutoffs = [float(arg)]
    elif setting == '-cutoffs':
        l = arg.split(',')
        cutoffs = [float(e) for e in l]
    elif setting == '-max_tree_count':
        max_tree_count = int(arg)
    else:
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

if inputtrees == None:
    raise Exception("Need a set of trees on which to run!")

count_trees = load_trees.get_trees(file=counttrees, max_count=max_tree_count)
input_trees = load_trees.get_trees(file=inputtrees, max_count=max_tree_count)

rnn_enron.DEBUG_PRINT = False

rng = RandomState(1234)
rng = RandomState(random_seed)

traintimer.begin()
word_counts = inference_enron.get_word_counts(count_trees)
(yes_weights,
 no_weights) = inference_enron.get_weights(word_counts)
only_sensitive = [t for t in input_trees if t.syntax == "4"]
sen_fraction = float(len(only_sensitive)) / len(input_trees)

for cutoff in cutoffs:
    indicators = inference_enron.get_indicators(cutoff, yes_weights)

    # Run on trees
    acc = inference_enron.get_accuracy(
        input_trees, indicators)
    (tp, fp, tn, fn) = inference_enron.get_confusion_numbers(
        input_trees, indicators)

    print("Cutoff: {:.2f}. On input data: acc {:.4f} % ({:.4f} %)".format(cutoff, acc, 1 - sen_fraction))
    print("Confusion Matrix (tp,fp,tn,fn)", tp, fp, tn, fn)
traintimer.end()

# Done
traintimer.report()
totaltimer.end().report()
