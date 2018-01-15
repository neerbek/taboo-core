# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:48:34 2017

@author: neerbek
"""

import sys
# sys.path.append("/home/neerbek/jan/phd/DLP/paraphrase/taboo-core")
import numpy
from numpy.random import RandomState

import rnn_enron
import server_rnn
import similarity.load_trees as load_trees
import ai_util

outputmodel = "model.txt"
inputmodel = None
traintrees = None
validtrees = None
testtrees = None
nx = 50
nh = 300
trainer = server_rnn.Trainer()
glove_path = "../../code/glove"
file_prefix = "save"
validation_frequency = 1
train_report_frequency = 1
balance_data = False
max_embedding_count = -1
max_tree_count = -1
output_running_model = -1
random_seed = 1234
cost_weight = numpy.array([1, 0, 0, 0, 1])
# trainer.learning_rate=0.001
# trainer.L1_reg=0.00
# trainer.L2_reg=0.0001
# trainer.n_epochs=1000
# trainer.batch_size=40
# trainer.retain_probability = 0.8
start_epoch = 0
use_RMS_cost = False
totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

def syntax():
    print(
        """syntax: train_model.py [-traintrees <trees>][-validtrees <trees>][-testtrees <trees>] [-inputmodel <model>] [-outputmodel <model>]
    [-nx <nx>] [-nh <nh>] [-lr <lr>] [-L1_reg <float>][-L2_reg <float>][-n_epochs <int>][-batch_size <int>][-valid_batch_size <int>]
    [-retain_probability <float>][-glove_path <glove_path>][-file_prefix <file_prefix>][-start_epoch <epoch>]
    [-validation_frequency <number>][-train_report_frequency <number>][-balance_data <True|False>]
    [-random_seed <int>][-max_embedding_count <int>][-max_tree_count <int>]
    [-sensitive_weight <number>][-non-sensitive_weight <number>][-use_RMS_cost]
    [-momentum <float>][-learn <gd|gdm|adagrad>
    [-output_running_model <int>]    # counts between output of the running model (save_<prefix>_running_<count>.txt,  -1 = no output
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv

# arglist = "train -inputtrees ../taboo-jan/functionality/201/trees_201_100_custom_0000_0250.txt -nx 100 -nh 100 -lr 0.01 -L1_reg 0.0001 -L2_reg 0 -n_epochs -1 -retain_probability 0.95 -batch_size 50 -glove_path ../code/glove/ -file_prefix save_screen1 -validation_frequency 300 -train_report_frequency 120".split()
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
    elif setting == '-lr':
        trainer.learning_rate = float(arg)
    elif setting == '-L1_reg':
        trainer.L1_reg = float(arg)
    elif setting == '-L2_reg':
        trainer.L2_reg = float(arg)
    elif setting == '-n_epochs':
        trainer.n_epochs = int(arg)
    elif setting == '-batch_size':
        trainer.batch_size = int(arg)
    elif setting == '-valid_batch_size':
        trainer.valid_batch_size = int(arg)
    elif setting == '-retain_probability':
        p = float(arg)
        if p > 1 or p < 0:
            raise Exception("retain_probability must be between 0 and 1", p)
        trainer.retain_probability = p
    elif setting == '-outputmodel':
        outputmodel = arg
    elif setting == '-traintrees':
        traintrees = arg
    elif setting == '-validtrees':
        validtrees = arg
    elif setting == '-testtrees':
        testtrees = arg
    elif setting == '-inputmodel':
        inputmodel = arg
    elif setting == '-glove_path':
        glove_path = arg
    elif setting == '-file_prefix':
        file_prefix = arg
    elif setting == '-start_epoch':
        start_epoch = int(arg)
    elif setting == '-validation_frequency':
        validation_frequency = ai_util.eval_expr(arg)
    elif setting == '-train_report_frequency':
        train_report_frequency = ai_util.eval_expr(arg)
    elif setting == '-balance_data':
        balance_data = (arg.lower() in ("yes", "true", "t", "1"))
        print("balance_data=", balance_data)
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
    elif setting == '-momentum':
        trainer.mc = float(arg)
    elif setting == '-learn':
        trainer.learn = arg
    elif setting == '-output_running_model':
        output_running_model = int(arg)
    else:
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        elif setting == '-use_RMS_cost':
            use_RMS_cost = True
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

if traintrees == None or validtrees == None:
    raise Exception("Need a set of trees on which to train!")

train_trees = load_trees.get_trees(file=traintrees)
valid_trees = load_trees.get_trees(file=validtrees)
test_trees = load_trees.get_trees(file=testtrees)

rnn_enron.DEBUG_PRINT = False

rng = RandomState(1234)
state = server_rnn.State(
    max_embedding_count=max_embedding_count,
    nx=nx,
    nh=nh,
    rng=rng,
    glove_path=glove_path)

state.train_trees = train_trees
state.valid_trees = valid_trees
state.test_trees = test_trees

state.init_trees(trainer)

rng = RandomState(random_seed)

rnnWrapper = server_rnn.RNNWrapper(rng=rng, cost_weight=cost_weight)
rnnWrapper.rnn.cost = rnnWrapper.rnn.regressionLayer.cost_cross
if use_RMS_cost:
    rnnWrapper.rnn.cost = rnnWrapper.rnn.regressionLayer.cost_RMS
if inputmodel != None:
    rnnWrapper.load(inputmodel)

# Training
traintimer.begin()
trainer.train(
    state=state,
    rnnWrapper=rnnWrapper,
    file_prefix=file_prefix,
    n_epochs=trainer.n_epochs,
    rng=rng,
    epoch=start_epoch,
    validation_frequency=validation_frequency,
    train_report_frequency=train_report_frequency,
    output_running_model=output_running_model,
    balance_trees=balance_data)
traintimer.end()

rnn_enron.RNNTimers.getinputarraytimer.report()
rnn_enron.RNNTimers.looptimer.report()
rnn_enron.RNNTimers.appendtimer.report()
rnn_enron.RNNTimers.evaltimer.report()
rnn_enron.RNNTimers.evaltimer2.report()
server_rnn.Timers.randomtimer.report()
server_rnn.Timers.calltheanotimer.report()
server_rnn.Timers.totaltimer.report()
# Done
traintimer.report()
totaltimer.end().report()
print("***train completed")
