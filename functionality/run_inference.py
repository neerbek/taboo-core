# -*- coding: utf-8 -*-
"""
Created on Jun 28 2017

@author: neerbek
"""

import sys
# import os
# os.chdir("/home/jneerbek/jan/taboo/taboo-core")
from numpy.random import RandomState

import ai_util
import rnn_enron
import inference_enron
import similarity.load_trees as load_trees


counttrees = None
inputtrees = None
supportCutoffs = [9]
cutoffs = [0]
max_tree_count = -1
random_seed = 1234

traintimer = ai_util.Timer("Model time: ")
totaltimer = ai_util.Timer("Total time: ")
totaltimer.begin()


def syntax():
    print(
        """syntax: run_inference.py [-counttrees <trees>][-inputtrees <trees>]
        [-cutoff <float>][-max_tree_count <int>][-supportCutoff <int>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
# print("WARN: using debug input!")
# arglist = "../taboo-jan/functionality/run_inference.py -counttrees ../taboo-core/functionality/201/train_custom250_random.txt -inputtrees ../taboo-jan/functionality/201/test_custom250_random.txt -cutoffs 0.8".split()
# arglist = "run_inference.py -counttrees /home/jneerbek/jan/ProjectsData/phd/DLP/Monsanto/data/trees/20191015c/trees0.zip$train_manual_sensitive.txt -inputtrees /home/jneerbek/jan/ProjectsData/phd/DLP/Monsanto/data/trees/20191015c/trees0.zip$dev_manual_sensitive.txt -cutoffs 0.1,0.6,0.65,0.7,0.75,0.78,0.79,0.795,0.8,0.805,0.81,0.82,0.83,0.835,0.84,0.8425,0.845,0.8475,0.85,0.855,0.86,0.9,1.0 -supportCutoff 0".split()
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

    setting
    next_i = i + 2
    if setting == '-inputtrees':
        inputtrees = arg
    elif setting == '-counttrees':
        counttrees = arg
    elif setting == '-cutoff':
        cutoffs = [float(arg)]
    elif setting == '-supportCutoff':
        supportCutoffs = [int(arg)]
    elif setting == '-supportCutoffs':
        splits = arg.split(',')
        supportCutoffs = [int(e) for e in splits]
    elif setting == '-cutoffs':
        splits = arg.split(',')
        cutoffs = [float(e) for e in splits]
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

rng = RandomState(random_seed)

traintimer.begin()
word_counts = inference_enron.get_word_counts(count_trees)
only_sensitive = [t for t in input_trees if t.syntax == "4" or t.syntax == "1"]
sen_fraction = float(len(only_sensitive)) / len(input_trees)

bestOutput = None
bestAcc = None
bestIndicators = None
cutoff = cutoffs[0]
for supportCutoff in supportCutoffs:
    (yes_weights,
     no_weights) = inference_enron.get_weights(word_counts, supportCutoff=supportCutoff)
    for cutoff in cutoffs:
        confidence = cutoff
        weights = yes_weights
        indicators = inference_enron.get_indicators(cutoff, yes_weights)

        # Run on trees
        ttrees = input_trees
        acc = inference_enron.get_accuracy(
            input_trees, indicators)
        (tp, fp, tn, fn) = inference_enron.get_confusion_numbers(
            input_trees, indicators)
        output = "SupportCutoff: {} Cutoff: {:.3f}. On input data: acc {:.4f} % ({:.4f} %)".format(supportCutoff, cutoff, acc, 1 - sen_fraction)
        output += "\nConfusion Matrix (tp,fp,tn,fn) {} {} {} {}".format(tp, fp, tn, fn)
        print(output)

        if bestAcc is None or acc > bestAcc:
            bestOutput = output
            bestAcc = acc
            bestIndicators = indicators

traintimer.end()

if bestOutput is not None:
    print()
    print("Best: ", bestOutput)
    print("sensitive keywords: {}".format(bestIndicators))


# Done
traintimer.report()
totaltimer.end().report()
