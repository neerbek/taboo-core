#  -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:48:34 2017

@author: neerbek
"""

import numpy
# os.chdir("..")
import sys
from numpy.random import RandomState

import theano
import theano.tensor as T

import ai_util
import rnn_enron
import server_rnn
import similarity.load_trees as load_trees
import LogFileReader

inputmodel = None
inputtrees = None
nx = 50
nh = 300
trainer = server_rnn.Trainer()
glove_path = "../../code/glove"
max_embedding_count = -1
max_tree_count = -1
random_seed = 1234
cost_weight = numpy.array([1, 0, 0, 0, 1])
# trainer.learning_rate=0.001
# trainer.L1_reg=0.00
# trainer.L2_reg=0.0001
# trainer.n_epochs=1000
# trainer.batch_size=40
# trainer.retain_probability = 0.8
retain_probabilities = [0.8]
start_epoch = 0
totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Eval time: ")
totaltimer.begin()
threshold = None
min_sentence_len = -1

def syntax():
    print("""syntax: run_model.py [-inputtrees <trees>] [-inputmodel <model>]
    [-nx <nx>] [-nh <nh>][-L1_reg <float>][-L2_reg <float>][-n_epochs <int>][-batch_size <int>]
    [-retain_probabilities <float>[,<float>,...][-glove_path <glove_path>][-start_epoch <epoch>]
    [-threshold <float>[,<float>...]] [-min_sentence_len <int>]
    [-random_seed <int>][-max_embedding_count <int>][-max_tree_count <int>]
    [-sensitive_weight <number>][-non-sensitive_weight <number>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
# arglist = "run_model -inputtrees ../taboo-jan/functionality/201/train_custom250_random.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probabilities 0.8 -batch_size 80 -glove_path ../code/glove/ -inputmodel save_exp56_running.txt -min_sentence_len 5 -max_embedding_count -1".split(" ")
# arglist = "run_model -inputtrees ../taboo-jan/functionality/201/train_full_random.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probabilities 0.8 -batch_size 80 -glove_path ../code/glove/ -inputmodel save_exp56_running.txt -min_sentence_len 5 -max_embedding_count -1".split(" ")
# here you can insert manual arglist if needed
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
    if setting == '-nx':
        nx = int(arg)
    elif setting == '-nh':
        nh = int(arg)
    elif setting == '-L1_reg':
        trainer.L1_reg = float(arg)
    elif setting == '-L2_reg':
        trainer.L2_reg = float(arg)
    elif setting == '-batch_size':
        trainer.batch_size = int(arg)
    elif setting == '-retain_probabilities':
        retain_probabilities = []
        l = arg.split(',')
        for e in l:
            p = float(e)
            if p > 1 or p < 0:
                raise Exception("retain_probability must be between 0 and 1", p)
            retain_probabilities.append(p)
    elif setting == '-inputtrees':
        inputtrees = arg
    elif setting == '-inputmodel':
        inputmodel = arg
    elif setting == '-glove_path':
        glove_path = arg
    elif setting == '-min_sentence_len':
        min_sentence_len = int(arg)
    elif setting == '-threshold':
        e = arg.split(",")
        threshold = []
        for a in e:
            threshold.append(float(a))
    elif setting == '-random_seed':
        random_seed = int(arg)
    elif setting == '-max_embedding_count':
        max_embedding_count = int(arg)
    elif setting == '-max_tree_count':
        max_tree_count = int(arg)
    elif setting == '-sensitive_weight':
        cost_weight[4] = cost_weight[4] * float(arg)
    elif setting == '-non-sensitive_weight':
        cost_weight[0] = cost_weight[0] * float(arg)
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
    raise Exception("Need a set of trees on which to train!")
if glove_path == None:
    raise Exception("Need a path to embeddings<!")

# min_sentence_lenx = -1
print("loading " + inputtrees)
input_trees = load_trees.get_trees(file=inputtrees, max_count=max_tree_count)
all_trees = input_trees
if min_sentence_len > -1:
    res = []
    for t in input_trees:
        if load_trees.count_nodes(t) > min_sentence_len:
            res.append(t)
    print("Removing short trees. Previous {} New length {}".format(len(input_trees), len(res)))
    input_trees = res
valid_trees = load_trees.get_trees(file=inputtrees, max_count=100)
test_trees = load_trees.get_trees(file=inputtrees, max_count=100)

rnn_enron.DEBUG_PRINT = False

rng = RandomState(1234)
state = server_rnn.State(
    max_embedding_count=max_embedding_count,
    nx=nx,
    nh=nh,
    rng=rng,
    glove_path=glove_path)

state.train_trees = input_trees
state.valid_trees = valid_trees
state.test_trees = test_trees

state.init_trees(trainer)

rng = RandomState(random_seed)

rnnWrapper = server_rnn.RNNWrapper(rng=rng, cost_weight=cost_weight)
if inputmodel != None:
    rnnWrapper.load(inputmodel)

# Theano functions
cost = trainer.get_cost(rnnWrapper)
cost_model = trainer.get_cost_model(rnnWrapper, cost)
validation_model = trainer.get_validation_model(rnnWrapper)
confusion_matrix = trainer.get_confusion_matrix(rnnWrapper)

class ThresholdFunc:
    def __init__(self, threshold):
        self.threshold = threshold

    def calc_threshold_error(self, y, p_y_given_x):
        y_simple = T.argmax(y, axis=1)
        p_sensitive = p_y_given_x[:, 4]
        y_pred = T.switch(p_sensitive > self.threshold, 4, 0)
        if y_simple.ndim != y_pred.ndim:
            raise TypeError(
                'y_simple should have the same shape as self.y_pred',
                ('y_simple', y_simple.type, 'y_pred', y_pred.type)
            )
            # check if y is of the correct datatype
        if y_simple.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(y_pred, y_simple))
        else:
            raise NotImplementedError()


validation_models = []
if threshold == None:
    func_orig = theano.function(
        inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
        outputs=rnnWrapper.rnn.errors(rnnWrapper.y)
    )
    validation_models.append((-1, func_orig))
else:
    for t in threshold:
        print("t is ", t)
        t_func = ThresholdFunc(t)
        func = theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=t_func.calc_threshold_error(rnnWrapper.y, rnnWrapper.rnn.p_y_given_x)
        )
        validation_models.append((t, func))

# Evaluate model
traintimer.begin()
performanceMeasurer = server_rnn.PerformanceMeasurer()

for retain_probability in retain_probabilities:
    for (t, func) in validation_models:
        performanceMeasurer.measure_trees(
            input_trees=input_trees, batch_size=trainer.valid_batch_size,
            retain_probability=retain_probability,
            rnn=rnnWrapper.rnn,
            validate_model=func,
            cost_model=cost_model, confusion_matrix=confusion_matrix)
        print("retain_prop: {:.4f}. Threshold {:.4f},  On data set: ".format(retain_probability, t))
        # performanceMeasurer.report(msg=)
        LogFileReader.logTrain(LogFileReader.Train(cost=performanceMeasurer.cost, nodeAccuracy=performanceMeasurer.root_acc, nodeCount=performanceMeasurer.total_root_nodes), epoch=-1, dataSetName="inputtrees (root)")
        cm = performanceMeasurer.total_confusion_matrix
        print("Confusion Matrix (tp,fp,tn,fn)", cm[0], cm[1], cm[2], cm[3])

traintimer.end()

# Done
traintimer.report()
totaltimer.end().report()
