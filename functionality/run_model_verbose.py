#  -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:48:34 2017

@author: neerbek
"""

import numpy
import sys
from numpy.random import RandomState

import theano
import theano.tensor as T

import ai_util
import rnn_enron
import server_rnn
import similarity.load_trees as load_trees
import confusion_matrix

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
retain_probability = 0.8
max_count = -1
totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Eval time: ")
totaltimer.begin()
output_embeddings = False
runOnAllNodes = False

def syntax():
    print("""syntax: run_model_verbose.py [-inputtrees <trees>] [-inputmodel <model>]
    [-nx <nx>] [-nh <nh>][-L1_reg <float>][-L2_reg <float>][-n_epochs <int>][-batch_size <int>]
    [-retain_probability <float>][-glove_path <glove_path>][-start_epoch <epoch>]
    [-random_seed <int>][-max_embedding_count <int>][-max_tree_count <int>]
    [-sensitive_weight <number>][-non-sensitive_weight <number>]
    [-max_count <int>][-output_embeddings][-runOnAllNodes]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
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
    elif setting == '-retain_probability':
        retain_probability = float(arg)
    elif setting == '-inputtrees':
        inputtrees = arg
    elif setting == '-inputmodel':
        inputmodel = arg
    elif setting == '-glove_path':
        glove_path = arg
    elif setting == '-random_seed':
        random_seed = int(arg)
    elif setting == '-max_embedding_count':
        max_embedding_count = int(arg)
    elif setting == '-max_tree_count':
        max_tree_count = int(arg)
    elif setting == '-max_count':
        max_count = int(arg)
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
        elif setting == '-output_embeddings':
            output_embeddings = True
        elif setting == '-runOnAllNodes':
            runOnAllNodes = True
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

input_trees = load_trees.get_trees(file=inputtrees, max_count=max_tree_count)
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
# confusion_matrix = trainer.get_confusion_matrix(rnnWrapper)

class VerboseValidationLogger:
    def __init__(self, rnnWrapper):
        self.log = []
        self.validation_model = theano.function(
            inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
            outputs=self.calc_threshold_error(rnnWrapper.y, rnnWrapper.rnn.p_y_given_x, rnnWrapper.rnn.reluLayer.output)
        )
        self.count = 0

    def calc_threshold_error(self, y, p_y_given_x, reluLayer_output):
        y_simple = T.argmax(y, axis=1)
        p_sensitive = p_y_given_x[:, 4]  # prob of being sensitive
        if y_simple.ndim != p_sensitive.ndim:
            raise TypeError(
                'y_simple should have the same shape as self.y_pred',
                ('y_simple', y_simple.type, 'y_pred', p_sensitive.type)
            )
            # check if y is of the correct datatype
        if y_simple.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred = T.argmax(p_y_given_x, axis=1)
            is_accurate = 1 - T.neq(y_pred, y_simple)
            return [y_simple, is_accurate, p_sensitive, reluLayer_output]  # list of arrays
        else:
            raise NotImplementedError()

    def wrapper(self, x_roots, y_roots, z_roots, trees):
        (truth_val, is_accurate, p_sensitive, reluLayer_output) = self.validation_model(x_roots, y_roots, z_roots)
        if len(x_roots) != len(trees):
            raise Exception("Count of trees are not equal {}, {}".format(len(x_roots), len(trees)))
        if len(x_roots) != len(is_accurate):
            raise Exception("Lengths of is_accurate are not equal {}, {}".format(len(x_roots), len(is_accurate)))
        if len(x_roots) != len(reluLayer_output):  # 2-dim array
            raise Exception("Lengths of reluLayer_output are not equal {}, {}".format(len(x_roots), len(reluLayer_output)))
        for i in range(len(trees)):
            node_count = load_trees.count_nodes(trees[i])
            text = load_trees.output_sentence(trees[i])
            tree_str = load_trees.output(trees[i])
            # (setenv "PYTHONPATH" "/home/neerbek/jan/phd/DLP/paraphrase/taboo-core/;/home/neerbek/jan/phd/DLP/paraphrase/taboo-jan/functionality")
            # (getenv "PYTHONPATH")
            entry = [truth_val[i], is_accurate[i], p_sensitive[i], node_count, text, tree_str, reluLayer_output[i]]
            self.log.append(entry)

    def report(self, max_count=100):
        if max_count == -1:
            max_count = len(self.log)
        max_count = min(max_count, len(self.log))
        header = confusion_matrix.EMBEDDING_FILE_HEADER_BASE
        if output_embeddings:
            header = confusion_matrix.EMBEDDING_FILE_HEADER_FULL
        print(header)
        for i in range(max_count):
            entry = self.log[i]
            msg = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i, entry[0], entry[1], entry[2], entry[3], entry[4], entry[5])
            if output_embeddings:
                arr = "["
                for e in entry[6]:
                    arr += str(e) + ","
                arr = arr[:-1] + "]"
                msg += "\t" + arr
            print(msg)


logger = VerboseValidationLogger(rnnWrapper)
verboseValidationModel = logger.wrapper

# validationModel = theano.function(
#     inputs=[rnnWrapper.x, rnnWrapper.y, rnnWrapper.z],
#     outputs=rnnWrapper.rnn.errors(rnnWrapper.y)
# )


# Evaluate model
traintimer.begin()
performanceMeasurer = server_rnn.PerformanceMeasurer()

performanceMeasurer.measure_roots(
    input_trees=input_trees, batch_size=trainer.valid_batch_size,
    retain_probability=retain_probability,
    rnn=rnnWrapper.rnn,
    measure_wrapper=verboseValidationModel, measureRoots=(not runOnAllNodes))

logger.report(max_count)

traintimer.end()

# Done
traintimer.report()
totaltimer.end().report()
