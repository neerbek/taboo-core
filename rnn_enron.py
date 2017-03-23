# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:49:15 2017

@author: neerbek
"""

# start spyder with: OMP_NUM_THREADS=3 spyder3
import os
import io
import sys
#import time
from datetime import datetime
import time

import numpy
from numpy.random import RandomState

import theano
import theano.tensor as T

import similarity.load_trees as load_trees
import deeplearning_tutorial.rnn4 as rnn
#reload(rnn)
from ai_util import Timer
from StatisticTextParser import StatisticTextParser

MAX_SENTENCE_LENGTH=600      #set lower if you need parsing to happen faster
DEBUG_PRINT_VERBOSE=False
DEBUG_PRINT = True


class Timers:
    def __init__(self):
        self.totalTimer = Timer("total time")
        self.gettreeTimer = Timer("gettree time")
        self.nltkTimer = Timer("nltk time")
        self.normTimer = Timer("norm parse time")
        self.ckyTimer = Timer("CKY time")
        self.timers = [self.totalTimer, self.gettreeTimer, self.nltkTimer, self.normTimer, self.ckyTimer]
        self.lastReport = time.time()
    def end(self):
        for t in self.timers:
            if t.start!=None:
                t.end()
        
    def report(self, min_seconds=-1):
        t = time.time()
        if (t-self.lastReport <min_seconds):
            return False
        self.lastReport = t
        self.ckyTimer.report()
        self.totalTimer.report()
        #for t in self.timers:
        #    t.report()
        return True
        
def add_node(tree, child):
    t = load_trees.Node(None)
    t.left = child
    if child.parent is not None:
        raise Exception("child tree already have parent")
    child.parent = t
    if tree.number_of_children()>1:
        raise Exception("No room to add child")
    tree.right = t
    t.parent=tree
    return t
    
class ParserStatistics:
    def __init__(self):
        self.sentences = 0
        self.splits = 0
        self.emptySubTrees = 0
        self.emptySentenceTrees = 0

def get_nltk_parsed_tree_from_sentence(l2, parser, timers, parserStatistics):
    fn = "get_nltk_parsed_tree_from_sentence"   #function name
    parserStatistics.sentences += 1
    l2 = load_trees.escape_sentence(l2)
    w = parser.tokenizer.tokenize(l2)
    l2_copy = None
    try:
        l2_copy = ' '.join(w)   #normalize whitespaces
    except TypeError:
        print("failed to join w: {}".format(w))
        raise
    l2_copy = l2_copy.replace(' . ','. ').replace(' , ',', ').replace(' ; ','; ').replace(' : ',': ')
    l2_copy = l2_copy.replace(' .\n','.\n').replace(' ,\n',',\n').replace(' ;\n',';\n').replace(' :\n',':\n')
    if len(l2_copy)>1 and l2_copy[-2] == ' ':
        if l2_copy[-1]=='.' or l2_copy[-1]==',' or l2_copy[-1]==';' or l2_copy[-1]==':':
            l2_copy = l2_copy[:-2] + l2_copy[-1]
    l2 = l2.strip()
    trees=[]
    if len(l2)>MAX_SENTENCE_LENGTH and DEBUG_PRINT_VERBOSE:
        print("Going to split long sentence: {}".format(len(l2)))
        print(l2)
    subsentence_skipped=False
    while len(l2)>0:
        tmp = None
        if len(l2)>MAX_SENTENCE_LENGTH:
            index = l2.find(' ', int(MAX_SENTENCE_LENGTH/2))
            if index!=-1:
                parserStatistics.splits += 1
                if DEBUG_PRINT:
                    print("Splitting long sentence: {}".format(len(l2)))
                tmp = l2[:index]
                l2 = l2[index+1:]
        if tmp is None:
            tmp = l2
            l2 = ""
        tmp = tmp.strip()
        tree = None
        try:
            timers.gettreeTimer.begin()
            tree = parser.get_tree(tmp, timers)
            timers.gettreeTimer.end()
        except Exception:
            timers.gettreeTimer.end()
            raise
        if tree is not None:
            trees.append(tree)
        else:
            parserStatistics.emptySubTrees += 1
            print("Warning: No tree obtained from parser. Sentence was: " + tmp)
            #continue here yields problems as the final tree will generate a different tree
            subsentence_skipped=True
            continue
    if len(trees)==0:
        parserStatistics.emptySentenceTrees += 1
        return None
    root = None
    if len(trees)==1:
        root = trees[0]
    else:
        tree = load_trees.Node(None)
        root = tree
        tree.left = trees[0]
        trees[0].parent=tree
        i = 1
        while i<len(trees):
            if i+1<len(trees):
                tree = add_node(tree, trees[i])
            else:
                tree.right = trees[i]
                trees[i].parent=tree
            i += 1
    if not subsentence_skipped:
        #we can only compare if we have skipped no subsentences
        l3 = load_trees.output_sentence(root)    
        l3 = l3.strip()
        if l2_copy!=l3: #final sentence must be the same
            raise Exception(fn + " marshall and unmarshalling differs" + "\n" + l2_copy + "\n" + l3)
    if root==None:
        parserStatistics.emptySentenceTrees += 1
    return root

def get_nltk_parsed_trees_from_list(lines):
    timers = Timers()
    count = 0
    none_count = 0
    trees = []
    parser = StatisticTextParser()
    fn = "nltk_parser"   #function name
    #f = io.open(file,'r',encoding='utf8')
    parserStatistics = ParserStatistics()
    for line in lines:
        timers.totalTimer.begin()
        line = line[:-1]   #strips newline. But consider: http://stackoverflow.com/questions/509446/python-reading-lines-w-o-n
        orig_tree = load_trees.Node(None)
        #print("Line is " + line)
        i = load_trees.parse_line(line, 2, orig_tree)
        if i<len(line)-1: #Not all of line parsed
            timers.end()
            raise Exception(fn + " parsing line failed. There was more than one tree in the line. {}".format(i))
        l2 = load_trees.output_sentence(orig_tree)
        l2 = l2.replace('\xa0',' ')   #some leftover non ascii codes xa0=line feed
        tree=None
        try:
            tree = get_nltk_parsed_tree_from_sentence(l2, parser, timers, parserStatistics)
        except Exception:
            timers.end()
            print("error for line " + l2)
            raise
        if tree is None:
            timers.totalTimer.end()
            count +=1
            none_count += 1
            continue
        tree.replace_nodenames(orig_tree.syntax)
        trees.append(tree)
        count +=1
        timers.totalTimer.end()
        if DEBUG_PRINT and count%1 == 0:
            print("Extracted: {} None count: {}".format(count, none_count))
            timers.report()
            sys.stdout.flush()
    print(fn + " done. Count={}".format(count))
    return (parserStatistics, trees)

file = "trees/train.txt"
def get_nltk_parsed_trees_from_file(file, start_from = -1, max_count=-1):
    lines =[]
    #f = io.open(file,'r',encoding='utf8')
    with io.open(file,'r',encoding='utf8') as f:        
        for line in f:
            lines.append(line)
        #line = lines[3]
        #line = lines[76]
        #line
    if start_from>-1:
        lines = lines[start_from:]
    if max_count>-1:
        lines = lines[:max_count]
    (parserStatistics, trees) = get_nltk_parsed_trees_from_list(lines)
    return trees

def get_word_embeddings(file, rng, max_count=-1):
    count = 0
    fn = "EmbeddingReader "   #function name
    res = {}
    with io.open(file,'r',encoding='utf8') as f:
        splitcount = -1
        for line in f:
            #line ="the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457"            
            if count == max_count:
                break
            e = line.split()
            if splitcount==-1:
                splitcount = len(e)
            if splitcount!=len(e):
                raise Exception(fn + "Wrong number of splits was: {}, expected: {}".format(len(e), splitcount))
            if e[0].lower() != e[0]:  #islower fails for ","
                raise Exception("word embedding was expected to lower case: " + e[0])
            w = e[0]            
            arr = numpy.array([float(it) for it in e[1:]])   #.reshape(1,50)
            res[w] = arr
            count +=1
            if count%60000 == 0:
                print(fn + "extracted: ", count)
    if len(res)>0:
        arr = next(iter(res.values()))  #get "random" element
        nx = len(arr)
        unknown = rng.uniform(-1, 1, size=nx)
        res[UNKNOWN_WORD] = unknown
    print(fn + "done. Count={}".format(count))
    return res

class NodeCounter:
    def __init__(self):
        self.node_count = 0
        self.word_count = 0
        self.unknown_count = 0
    def incNode(self):
        self.node_count += 1
    def incWord(self):
        self.word_count += 1
    def incUnknown(self):
        self.unknown_count += 1
    def add(self, nodeCounter):
        self.node_count += nodeCounter.node_count
        self.word_count += nodeCounter.word_count
        self.unknown_count += nodeCounter.unknown_count
        
UNKNOWN_WORD = "*unknown*"
def updateWordVectors(node, lookup_table, nodeCounter):
    # assumes is_binary and has_only_words_at_leafs
    nodeCounter.incNode()
    if node.is_leaf():
        nodeCounter.incWord()
        #TODO: use lower
        #word = node.word.lower()
        word = node.word
        if word == "-LRB-":
            word = "("
        elif word == "-RRB-":
            word = ")"
        elif word == "-LSB-":
            word = "("
        elif word == "-RSB-":
            word = ")"
        elif word == "-LCB-":
            word = "("
        elif word == "-RCB-":
            word = ")"

        if word in lookup_table:
            node.representation = lookup_table[word]
        else:
            nodeCounter.incUnknown()
            node.representation = lookup_table[UNKNOWN_WORD]
    else:
        updateWordVectors(node.left, lookup_table, nodeCounter)
        updateWordVectors(node.right, lookup_table, nodeCounter)

def setTreeLabel(node, label):
    if node is None:
        return
        
    node.label = label
    setTreeLabel(node.left, label)
    setTreeLabel(node.right, label)
    
def initializeTrees(trees, LT):
    totalCounter = NodeCounter()
    for tree in trees:
        nodeCounter = NodeCounter()
        updateWordVectors(tree, LT, nodeCounter)
        totalCounter.add(nodeCounter)
        label = numpy.zeros(5)   #TODO: remove hardcoded constant
        if tree.syntax=="4":
            label[4] = 1
        elif tree.syntax=="0":
            label[0] = 1
        else:
            raise Exception("tree does not have correct syntax label")
        setTreeLabel(tree, label)
    print("Done with tree. Saw {} nodes, {} words and {} unknowns. Unknown ratio is {}".format(totalCounter.node_count, totalCounter.word_count, totalCounter.unknown_count, (totalCounter.unknown_count + 0.0)/totalCounter.word_count))

#def nodeForward(node, update_count, reg):
#    # assumes is_binary and has_only_words_at_leafs
#    if node.is_leaf():
#        return  #don't update words representations
#    else:
#        if node.update_count < update_count:
#            nodeForward(node.left, update_count, reg)
#            nodeForward(node.right, update_count, reg)                
#            node.representation = reg.get_representation(node.left.representation, node.right.representation)
#            node.update_count = update_count

#329225 + 56255
#22017+3568
#96141 + 16560
#79063+13774
## expected: lookups 385480 unknowns 92837
## Unknown lookup ratio: 0.240835
#
#t = train_trees[0]
#t.label
#t.is_leaf()
#t = t.left
#t.representation
######################
# BUILD ACTUAL MODEL #
######################

class Evaluator:
    SIZE=None     #initialize this
    HIDDEN_SIZE=300
    RES_SIZE=5
    hidden_rep = numpy.zeros(HIDDEN_SIZE)
    leaf_rep = None
    def set_size(SIZE):
        Evaluator.SIZE = SIZE
        Evaluator.leaf_rep = numpy.zeros(Evaluator.SIZE)
        
    def __init__(self, reg):
        self.W = reg.reluLayer.W.eval()
        self.b = reg.reluLayer.b.eval()
        self.in_rep = None
        #self.in_rep = numpy.random.uniform(-1, 1, size=100)
        self.reg=reg
    def get_representation(self, left, right):
        #print("shape is", left_in.shape)
        #        for i in range(Evaluator.SIZE):
        #            self.in_rep[0,i] = left_in[0,i]
        #            self.in_rep[0,Evaluator.SIZE + i] = right_in[0,i]           
        #lin_output = T.dot(X, self.W) + self.b
        #self.output = T.nnet.relu(lin_output)
        l = []
        if left.is_leaf():
            l.append(Evaluator.hidden_rep)
            l.append(left.representation)
        else:
            l.append(left.representation)
            l.append(Evaluator.leaf_rep)
        if right.is_leaf():
            l.append(Evaluator.hidden_rep)
            l.append(right.representation)
        else:
            l.append(right.representation)
            l.append(Evaluator.leaf_rep)
        try:
            self.in_rep = numpy.concatenate(l)   #.reshape(1, 2*Evaluator.SIZE)
        except Exception:
            print("ups")
            raise
        lin = numpy.dot(self.in_rep, self.W) + self.b        
        lin = numpy.maximum(lin, 0, lin)
        #lin = right_in
        #        
        #        lin2 = reg.get_representation(left_in.reshape(1,50), right_in.reshape(1,50)).reshape(-1)
        #        if lin.shape!=lin2.shape:
        #            print("shapes wrong {} vs. {}".format(lin.shape, lin2.shape))
        #            raise Exception("Arrays not equally shaped!")
        #        if not numpy.allclose(lin, lin2):
        #            print("Arrays not equal", lin, lin2)
        #            raise Exception("Arrays not equal!")
        return lin

#def softmax(w):
#    #print("softmax: " , w.shape)
#    maxes = numpy.amax(w, axis=1)
#    maxes = maxes.reshape(maxes.shape[0], 1)
#    e = numpy.exp(w - maxes)
#    dist = e / numpy.sum(e, axis=1, keepdims=True)
#    return dist
#
#def theano_softmax():
#    x = T.dmatrix('x')
#    _y = T.nnet.softmax(x)
#    f = theano.function([x], _y)
#    return f
#    
##t = theano_softmax()
##i = numpy.random.randn(3,2)
##i = numpy.zeros((5,1))
##t(i)
##softmax(i)
#
#class CostEvaluator:
#    SIZE=50
#    MIN_VAR = 0.0000000001
#    def __init__(self, reg, L1_reg, L2_reg):
#        self.W_s = reg.regressionLayer.W.eval()
#        self.b_s = reg.regressionLayer.b.eval()
#        self.W = reg.reluLayer.W.eval()
#        self.b = reg.reluLayer.b.eval()
#        self.L1_reg = L1_reg
#        self.L2_reg = L2_reg
#    def get_representation(self, x):
#        lin = numpy.dot(x, self.W) + self.b        
#        lin = numpy.maximum(lin, 0, lin)
#        #        lin2 = reg.get_representation_x(x).reshape(-1)
#        #        if lin.shape!=lin2.shape:
#        #            print("shapes wrong {} vs. {}".format(lin.shape, lin2.shape))
#        #            raise Exception("Arrays not equally shaped!")
#        #        if not numpy.allclose(lin, lin2):
#        #            print("Arrays not equal", lin, lin2)
#        #            raise Exception("Arrays not equal!")
#        return lin
#    def get_output(self, x):
#        lin = self.get_representation(x)
#        dot = (numpy.dot(lin, self.W_s) + self.b_s).reshape(x.shape[0],5)
#        sm = softmax(dot)
#        return sm
#    def get_L1(self):
#        return abs(self.W).sum() + abs(self.W_s).sum()
#    def get_L2(self):
#        return (self.W ** 2).sum() + (self.W_s ** 2).sum()
#    def get_cost(self, x, y):
#        p_y_given_x = self.get_output(x)
#        cost =numpy.mean(0.5 * (p_y_given_x - y) **2)
#        cost += self.L1_reg *self.get_L1() + self.L2_reg*self.get_L2()
#        return cost
#    def numeric_gradient(self, x_in, y_in, params, index_i, index_j):
#        length = x_in.shape[0]
#        acc_cost = 0
#        params[index_i, index_j] += CostEvaluator.MIN_VAR
#        for i in range(length):
#            acc_cost += self.get_cost(x_in[i].reshape(1,x_in.shape[1]), y_in[i].reshape(1,y_in.shape[1]))
#        params[index_i, index_j] -= 2*CostEvaluator.MIN_VAR
#        for i in range(length):
#            acc_cost -= self.get_cost(x_in[i].reshape(1,x_in.shape[1]), y_in[i].reshape(1,y_in.shape[1]))
#        acc_cost /= 2*CostEvaluator.MIN_VAR
#        return acc_cost
#        
            

class Ctxt:
    evaltimer = Timer("eval timer")
    appendtimer = Timer("append timer")

def addNodeRepresentations(reg, node, x_val, y_val, evaluator):
    if node == None:
        return
    if node.is_leaf():
        return
        
    addNodeRepresentations(reg, node.left, x_val, y_val, evaluator)
    addNodeRepresentations(reg, node.right, x_val, y_val, evaluator)
    
    #Ctxt.evaltimer.begin()
    node.representation = evaluator.get_representation(node.left, node.right)
    #Ctxt.evaltimer.end()
    
    #Ctxt.appendtimer.begin()
    y_val.append(node.label)
    x_val.append(evaluator.in_rep) 
    # we need to reshape in_rep but this will add 150% to running time
    #Ctxt.appendtimer.end()

def getInputArrays(reg, trees, evaluator):
    list_x = []
    list_y = []
    list_root_indexes = []
    for t in trees:
        addNodeRepresentations(reg, t, list_x, list_y, evaluator)
        list_root_indexes.append(len(list_x)-1) #root is added last
    #train_set_x = theano.shared(numpy.asarray(train_set_x), borrow = True)
    #train_set_y = theano.shared(numpy.asarray(train_set_y), borrow = True)        
    #list_x = [e.reshape(-1) for e in list_x]
    x_val = numpy.concatenate(list_x,axis=0).reshape(-1,2*(Evaluator.SIZE + Evaluator.HIDDEN_SIZE))
    #x_val = numpy.vstack(list_x)  - shower by 1.1 sec
    y_val = numpy.concatenate(list_y,axis=0).reshape(-1,Evaluator.RES_SIZE)
    #x_val = numpy.asarray(list_x)
    #y_val = numpy.asarray(list_y)
    if x_val.shape != (len(list_x), 2*(Evaluator.SIZE + Evaluator.HIDDEN_SIZE)):
        raise Exception("error in numpy conversion of x, shape was {}".format(x_val.shape))
    if y_val.shape != (len(list_y), Evaluator.RES_SIZE):
        raise Exception("error in numpy conversion of y")
    return (list_root_indexes, x_val, y_val)


#
# numeric gradient
#
#W_grad_func = theano.function(
#    inputs=[x,y],
#    outputs=T.grad(cost, reg.regressionLayer.W)
#    )
#    
#minibatch_index=4
#trees = train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
#evaluator = Evaluator(reg)
#(x_val, y_val) = getInputArrays(reg, trees, evaluator)
#for i in range(50,100):
#    print(y_val[i])
#x_val = x_val[53:54]
#y_val = y_val[53:54]
#
#
#W_grad = W_grad_func(x_val, y_val)
#W_grad.shape  #(50,5)
#gradEvaluator = CostEvaluator(reg, L1_reg, L2_reg)
##gradEvaluator.get_representation(x_val)
##reg.reluLayer.output.eval({x:x_val})
##numpy.allclose(gradEvaluator.get_representation(x_val), reg.reluLayer.output.eval({x:x_val}))
##gradEvaluator.get_output(x_val)
##reg.regressionLayer.p_y_given_x.eval({x:x_val})
##gradEvaluator.get_cost(x_val,y_val)
##cost.eval({x:x_val,y:y_val})
##x_val = x_val.reshape(1,100)
##y_val = y_val.reshape(1,5)
##i=10
##j=1
##gradEvaluator.get_cost(x_val[0].reshape(1,x_val.shape[1]), y_val[0].reshape(1,y_val.shape[1]))
##gradEvaluator.numeric_gradient(x_val, y_val, gradEvaluator.W_s, i,j)
#length = x_val.shape[0]
#for index in range(length):
#    x_val1 = x_val[index].reshape(1,x_val.shape[1])
#    y_val1 = y_val[index].reshape(1,y_val.shape[1])
#    W_grad = W_grad_func(x_val1, y_val1)
#    for i in range(50):
#        for j in range(5):
#            if abs(gradEvaluator.numeric_gradient(x_val1, y_val1, gradEvaluator.W_s, i,j) - W_grad[i,j])>0.000001:
#                raise Exception("index {},{} is not close {},{}".format(i,j, gradEvaluator.numeric_gradient(x_val1, y_val1, gradEvaluator.W_s, i,j), W_grad[i,j]))
#    if index%100==0:
#        print("index:",index)
#print("all values passed")
#W_grad[i,j]
#gradEvaluator.get_L1()
#gradEvaluator.get_L2()
#gradEvaluator.L1_reg
#gradEvaluator.L2_reg
#reg.regressionLayer.W.eval()[16,j]

#
# END numeric gradient
#

def get_zeros(y_val):
    zeros = 0
    for e in y_val:
        if e[0]==0:
            zeros +=1
    return (zeros + 0.0) / len(y_val)
    
#n_epochs=1
train_times = 1
def run(n_epochs):
    validation_frequency = n_train_batches/2
    epoch = 0
    it = 0
    totaltimer = Timer("loop timer")
    traintimer = Timer("train timer")
    validtimer = Timer("validation timer")
    inputtimer = Timer("input timer")
    Ctxt.evaltimer = Timer("eval timer")
    Ctxt.appendtimer = Timer("append timer")
    while (n_epochs==-1 or epoch < n_epochs):
        totaltimer.begin()
        epoch += 1
        minibatch_index=0
        for minibatch_index in range(n_train_batches):
            trees = train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            evaluator = Evaluator(reg)
            inputtimer.begin()
            (roots, x_val, y_val) = getInputArrays(reg, trees, evaluator)
            inputtimer.end()
            traintimer.begin()
            minibatch_avg_cost = []
            for i in range(train_times):
                #z_val = rng.binomial(n=1, size=(974, 700), p=0.8)
                z_val = rng.binomial(n=1, size=(x_val.shape[0], Evaluator.HIDDEN_SIZE), p=retain_probability)
                minibatch_avg_cost.append(train_model(x_val, y_val, z_val))
            traintimer.end()
            minibatch_avg_cost = numpy.mean(minibatch_avg_cost)
            it += 1
            if it % validation_frequency == 0:
                validation_losses = []
                val_total_zeros = []
                val_root_losses = []
                val_root_zeros = []
                validation_cost = []
                evaluator = Evaluator(reg)
                #c = prng.randint(n_valid_batches, size=n_valid_batches/train_times)
                for i in range(n_valid_batches):
                    trees = valid_trees[i * valid_batch_size: (i + 1) * valid_batch_size]
                    inputtimer.begin()
                    (roots, x_val, y_val) = getInputArrays(reg, trees, evaluator)
                    inputtimer.end()
                    validtimer.begin()
                    z_val = numpy.ones(shape=(x_val.shape[0], Evaluator.HIDDEN_SIZE))
                    z_val = z_val * retain_probability
                    validation_losses.append(validate_model(x_val, y_val, z_val))
                    validation_cost.append(cost_model(x_val, y_val, z_val))
                    validtimer.end()
                    val_total_zeros.append(get_zeros(y_val))
                    x_roots = []
                    y_roots = []
                    for r in roots:
                        x_roots.append(x_val[r,:])
                        y_roots.append(y_val[r,:])
                    z_roots = numpy.ones(shape=(len(x_roots), Evaluator.HIDDEN_SIZE))
                    z_roots = z_roots * retain_probability
                    val_root_losses.append(validate_model(x_roots, y_roots, z_roots))
                    val_root_zeros.append(get_zeros(y_roots))
                val_total_acc = 1 - numpy.mean(validation_losses)
                val_root_acc = 1  - numpy.mean(val_root_losses)
                val_total_zeros = 1 - numpy.mean(val_total_zeros)
                val_root_zeros = 1 - numpy.mean(val_root_zeros)
                val_cost = numpy.mean(validation_cost)
                print("epoch {}. time is {}, minibatch {}/{}, validation total accuracy {:.4f} % ({:.4f} %) validation cost {:.6f}, val root acc {:.4f} % ({:.4f} %)".format(
                        epoch, datetime.now().strftime('%d-%m %H:%M'),
                        minibatch_index + 1, n_train_batches, val_total_acc*100., val_total_zeros*100.,
                        val_cost*1.0, val_root_acc*100., val_root_zeros*100.
                    ))
                #have to mult by 1.0 to convert minibatch_avg_cost from theano to python variables
        totaltimer.end().report()
        traintimer.report(totaltimer.count)
        validtimer.report(totaltimer.count)
        inputtimer.report(totaltimer.count)
        Ctxt.evaltimer.report(totaltimer.count)
        Ctxt.appendtimer.report(totaltimer.count)



#with inner timers
#loop timer: Number of updates: 3. Total time: 60.98. Average time: 20.3266 sec
#train timer: Number of updates: 3. Total time: 1.85. Average time: 0.6156 sec
#validation timer: Number of updates: 3. Total time: 1.11. Average time: 0.3704 sec
#input timer: Number of updates: 3. Total time: 57.88. Average time: 19.2929 sec
#eval timer: Number of updates: 3. Total time: 31.38. Average time: 10.4599 sec
#append timer: Number of updates: 3. Total time: 12.21. Average time: 4.0691 sec

#with inner timers and with youtube
#loop timer: Number of updates: 3. Total time: 62.93. Average time: 20.9770 sec
#train timer: Number of updates: 3. Total time: 2.34. Average time: 0.7811 sec
#validation timer: Number of updates: 3. Total time: 1.70. Average time: 0.5674 sec
#input timer: Number of updates: 3. Total time: 58.69. Average time: 19.5620 sec
#eval timer: Number of updates: 3. Total time: 31.90. Average time: 10.6319 sec
#append timer: Number of updates: 3. Total time: 12.16. Average time: 4.0537 sec

#with inner timers, youtube and reuse of concat
#loop timer: Number of updates: 3. Total time: 58.31. Average time: 19.4381 sec
#train timer: Number of updates: 3. Total time: 2.45. Average time: 0.8175 sec
#validation timer: Number of updates: 3. Total time: 1.73. Average time: 0.5766 sec
#input timer: Number of updates: 3. Total time: 53.93. Average time: 17.9779 sec
#eval timer: Number of updates: 3. Total time: 32.56. Average time: 10.8538 sec
#append timer: Number of updates: 3. Total time: 6.44. Average time: 2.1483 sec

#with inner timers, youtube and reuse concat, with of my own impl of concat (-r2484)
#loop timer: Number of updates: 3. Total time: 175.29. Average time: 58.4284 sec
#train timer: Number of updates: 3. Total time: 2.38. Average time: 0.7938 sec
#validation timer: Number of updates: 3. Total time: 1.77. Average time: 0.5900 sec
#input timer: Number of updates: 3. Total time: 170.95. Average time: 56.9832 sec
#eval timer: Number of updates: 3. Total time: 150.59. Average time: 50.1976 sec
#append timer: Number of updates: 3. Total time: 6.75. Average time: 2.2516 sec

#with inner timers, NO youtube and reuse of concat, using a run function
#loop timer: Number of updates: 3. Total time: 54.82. Average time: 18.2734 sec
#train timer: Number of updates: 3. Total time: 1.89. Average time: 0.6312 sec
#validation timer: Number of updates: 3. Total time: 1.07. Average time: 0.3574 sec
#input timer: Number of updates: 3. Total time: 51.70. Average time: 17.2320 sec
#eval timer: Number of updates: 3. Total time: 30.92. Average time: 10.3056 sec
#append timer: Number of updates: 3. Total time: 6.59. Average time: 2.1975 sec


#without inner timers
#loop timer: Number of updates: 3. Total time: 52.40. Average time: 17.4660 sec
#train timer: Number of updates: 3. Total time: 1.76. Average time: 0.5871 sec
#validation timer: Number of updates: 3. Total time: 1.08. Average time: 0.3615 sec
#input timer: Number of updates: 3. Total time: 49.41. Average time: 16.4687 sec
#eval timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec

#without inner, but with youtube running
#loop timer: Number of updates: 3. Total time: 53.99. Average time: 17.9975 sec
#train timer: Number of updates: 3. Total time: 2.64. Average time: 0.8796 sec
#validation timer: Number of updates: 3. Total time: 1.89. Average time: 0.6307 sec
#input timer: Number of updates: 3. Total time: 49.21. Average time: 16.4044 sec
#eval timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec

#WITH inner, but with youtube running
#loop timer: Number of updates: 5. Total time: 93.78. Average time: 18.7563 sec
#train timer: Number of updates: 5. Total time: 3.76. Average time: 0.7512 sec
#validation timer: Number of updates: 5. Total time: 2.55. Average time: 0.5098 sec
#input timer: Number of updates: 5. Total time: 87.14. Average time: 17.4289 sec
#eval timer: Number of updates: 5. Total time: 52.33. Average time: 10.4653 sec
#append timer: Number of updates: 5. Total time: 10.56. Average time: 2.1110 sec
#num updates per epoch to eval (and append): 809276
#true time for append is estimated 1.7sec

#WITH inner, but with youtube running and without rehape in append (but then later on)
#loop timer: Number of updates: 3. Total time: 57.08. Average time: 19.0277 sec
#train timer: Number of updates: 3. Total time: 2.51. Average time: 0.8352 sec
#validation timer: Number of updates: 3. Total time: 1.74. Average time: 0.5792 sec
#input timer: Number of updates: 3. Total time: 52.64. Average time: 17.5462 sec
#eval timer: Number of updates: 3. Total time: 31.07. Average time: 10.3568 sec
#append timer: Number of updates: 3. Total time: 2.20. Average time: 0.7326 sec
#true time for append is estimated 0.3sec

#WITH inner, but with youtube running and without rehape in append, using concatenate to reshape
#loop timer: Number of updates: 3. Total time: 51.44. Average time: 17.1453 sec
#train timer: Number of updates: 3. Total time: 2.38. Average time: 0.7926 sec
#validation timer: Number of updates: 3. Total time: 1.54. Average time: 0.5118 sec
#input timer: Number of updates: 3. Total time: 47.33. Average time: 15.7768 sec
#eval timer: Number of updates: 3. Total time: 31.27. Average time: 10.4227 sec
#append timer: Number of updates: 3. Total time: 2.24. Average time: 0.7473 sec
#true time for append is estimated 0.3sec

#As above but using vstack
#loop timer: Number of updates: 3. Total time: 54.89. Average time: 18.2971 sec
#train timer: Number of updates: 3. Total time: 2.09. Average time: 0.6979 sec
#validation timer: Number of updates: 3. Total time: 1.36. Average time: 0.4547 sec
#input timer: Number of updates: 3. Total time: 51.22. Average time: 17.0746 sec
#eval timer: Number of updates: 3. Total time: 30.92. Average time: 10.3059 sec
#append timer: Number of updates: 3. Total time: 2.21. Average time: 0.7365 sec
#true time for append is estimated 0.3sec

#with eval inner, but not append, but with youtube running and without rehape in append, using concatenate to reshape 
#(i.e. as 2 above)
#loop timer: Number of updates: 3. Total time: 48.39. Average time: 16.1296 sec
#train timer: Number of updates: 3. Total time: 2.32. Average time: 0.7749 sec
#validation timer: Number of updates: 3. Total time: 1.68. Average time: 0.5615 sec
#input timer: Number of updates: 3. Total time: 44.20. Average time: 14.7330 sec
#eval timer: Number of updates: 3. Total time: 31.48. Average time: 10.4924 sec
#append timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec
#true time for append is estimated 0.3sec

#removing reshape from word vectors and internal reps
#loop timer: Number of updates: 4. Total time: 51.13. Average time: 12.7829 sec
#train timer: Number of updates: 4. Total time: 2.99. Average time: 0.7484 sec
#validation timer: Number of updates: 4. Total time: 1.98. Average time: 0.4962 sec
#input timer: Number of updates: 4. Total time: 45.87. Average time: 11.4685 sec
#eval timer: Number of updates: 4. Total time: 31.22. Average time: 7.8059 sec
#append timer: Number of updates: 4. Total time: 0.00. Average time: 0.0000 sec

#just doing concat in eval - ALL measures with youtube running!
#loop timer: Number of updates: 2. Total time: 15.80. Average time: 7.9016 sec
#eval timer: Number of updates: 2. Total time: 5.24. Average time: 2.6193 sec
#without eval timer, empty eval
#loop timer: Number of updates: 4. Total time: 17.88. Average time: 4.4692 sec
#eval timer: Number of updates: 4. Total time: 0.00. Average time: 0.0000 sec
#without eval timer, just doing concat in eval
#loop timer: Number of updates: 4. Total time: 23.79. Average time: 5.9466 sec
#eval timer: Number of updates: 4. Total time: 0.00. Average time: 0.0000 sec
#without eval timer using maximum
#loop timer: Number of updates: 2. Total time: 15.25. Average time: 7.6248 sec
#eval timer: Number of updates: 2. Total time: 0.00. Average time: 0.0000 sec
#without eval timer without maximum
#loop timer: Number of updates: 3. Total time: 26.11. Average time: 9.3633 sec
#eval timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec
#without eval timer full calc
#loop timer: Number of updates: 3. Total time: 33.70. Average time: 11.2329 sec
#eval timer: Number of updates: 3. Total time: 0.00. Average time: 0.0000 sec
#concat is approx 1.5 secs
#max is approx 1.7-1.9 secs
#dot is approx 3.6 secs

#With 2 validations per loop
#loop timer: Number of updates: 16. Total time: 63.59. Average time: 3.9744 sec
#train timer: Number of updates: 16. Total time: 9.05. Average time: 0.5654 sec
#validation timer: Number of updates: 16. Total time: 1.26. Average time: 0.0785 sec
#input timer: Number of updates: 16. Total time: 52.52. Average time: 3.2827 sec
#eval timer: Number of updates: 16. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 16. Total time: 0.00. Average time: 0.0000 sec
#with batch_size=225
#loop timer: Number of updates: 6. Total time: 23.50. Average time: 3.9173 sec
#train timer: Number of updates: 6. Total time: 3.50. Average time: 0.5826 sec
#validation timer: Number of updates: 6. Total time: 0.45. Average time: 0.0742 sec
#input timer: Number of updates: 6. Total time: 19.52. Average time: 3.2540 sec
#eval timer: Number of updates: 6. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 6. Total time: 0.00. Average time: 0.0000 sec
#with batch_size=9000
#loop timer: Number of updates: 43. Total time: 130.91. Average time: 3.0443 sec
#train timer: Number of updates: 43. Total time: 30.02. Average time: 0.6982 sec
#validation timer: Number of updates: 43. Total time: 0.00. Average time: 0.0000 sec
#input timer: Number of updates: 43. Total time: 100.58. Average time: 2.3390 sec
#eval timer: Number of updates: 43. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 43. Total time: 0.00. Average time: 0.0000 sec
#with batch_size=40, 1 validation per loop, using 2x input
#loop timer: Number of updates: 20. Total time: 89.01. Average time: 4.4507 sec
#train timer: Number of updates: 20. Total time: 14.55. Average time: 0.7275 sec
#validation timer: Number of updates: 20. Total time: 2.32. Average time: 0.1159 sec
#input timer: Number of updates: 20. Total time: 71.15. Average time: 3.5573 sec
#eval timer: Number of updates: 20. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 20. Total time: 0.00. Average time: 0.0000 sec
#same as above, n_hidden=300
#loop timer: Number of updates: 18. Total time: 454.86. Average time: 25.2701 sec
#train timer: Number of updates: 18. Total time: 140.17. Average time: 7.7871 sec
#validation timer: Number of updates: 18. Total time: 21.62. Average time: 1.2013 sec
#input timer: Number of updates: 18. Total time: 291.15. Average time: 16.1750 sec
#eval timer: Number of updates: 18. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 18. Total time: 0.00. Average time: 0.0000 sec
#same as above with 2 threads
#loop timer: Number of updates: 9. Total time: 174.65. Average time: 19.4058 sec
#train timer: Number of updates: 9. Total time: 46.30. Average time: 5.1447 sec
#validation timer: Number of updates: 9. Total time: 6.66. Average time: 0.7397 sec
#input timer: Number of updates: 9. Total time: 120.75. Average time: 13.4169 sec
#eval timer: Number of updates: 9. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 9. Total time: 0.00. Average time: 0.0000 sec
#same as above with 4 threads
#loop timer: Number of updates: 9. Total time: 170.51. Average time: 18.9454 sec
#train timer: Number of updates: 9. Total time: 37.70. Average time: 4.1888 sec
#validation timer: Number of updates: 9. Total time: 5.53. Average time: 0.6146 sec
#input timer: Number of updates: 9. Total time: 126.33. Average time: 14.0364 sec
#eval timer: Number of updates: 9. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 9. Total time: 0.00. Average time: 0.0000 sec
#same as above with 3 threads
#loop timer: Number of updates: 8. Total time: 144.67. Average time: 18.0831 sec
#train timer: Number of updates: 8. Total time: 32.59. Average time: 4.0735 sec
#validation timer: Number of updates: 8. Total time: 4.73. Average time: 0.5915 sec
#input timer: Number of updates: 8. Total time: 106.53. Average time: 13.3156 sec
#eval timer: Number of updates: 8. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 8. Total time: 0.00. Average time: 0.0000 sec
#same as above with 5 threads
#loop timer: Number of updates: 4. Total time: 81.60. Average time: 20.4009 sec
#train timer: Number of updates: 4. Total time: 18.82. Average time: 4.7042 sec
#validation timer: Number of updates: 4. Total time: 2.80. Average time: 0.7008 sec
#input timer: Number of updates: 4. Total time: 59.55. Average time: 14.8884 sec
#eval timer: Number of updates: 4. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 4. Total time: 0.00. Average time: 0.0000 sec
#adding dropout, threads=3
#loop timer: Number of updates: 10. Total time: 203.75. Average time: 20.3749 sec
#train timer: Number of updates: 10. Total time: 60.62. Average time: 6.0623 sec
#validation timer: Number of updates: 10. Total time: 6.49. Average time: 0.6494 sec
#input timer: Number of updates: 10. Total time: 135.57. Average time: 13.5569 sec
#eval timer: Number of updates: 10. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 10. Total time: 0.00. Average time: 0.0000 sec
    # Other laptop, as above unlimited (4) threads
#loop timer: Number of updates: 20. Total time: 976.28. Average time: 48.8142 sec
#train timer: Number of updates: 20. Total time: 306.07. Average time: 15.3033 sec
#validation timer: Number of updates: 20. Total time: 33.63. Average time: 1.6817 sec
#input timer: Number of updates: 20. Total time: 632.79. Average time: 31.6394 sec
#eval timer: Number of updates: 20. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 20. Total time: 0.00. Average time: 0.0000 sec
# Other laptop, as above 1 threads
#loop timer: Number of updates: 6. Total time: 290.21. Average time: 48.3689 sec
#train timer: Number of updates: 6. Total time: 108.51. Average time: 18.0855 sec
#validation timer: Number of updates: 6. Total time: 14.53. Average time: 2.4216 sec
#input timer: Number of updates: 6. Total time: 166.05. Average time: 27.6754 sec
#eval timer: Number of updates: 6. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 6. Total time: 0.00. Average time: 0.0000 sec
# Other laptop, as above 2 threads
#loop timer: Number of updates: 29. Total time: 1097.38. Average time: 37.8406 sec
#train timer: Number of updates: 29. Total time: 367.66. Average time: 12.6779 sec
#validation timer: Number of updates: 29. Total time: 43.24. Average time: 1.4909 sec
#input timer: Number of updates: 29. Total time: 681.31. Average time: 23.4933 sec
#eval timer: Number of updates: 29. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 29. Total time: 0.00. Average time: 0.0000 sec
# Other laptop, as above 3 threads
#loop timer: Number of updates: 50. Total time: 2466.92. Average time: 49.3385 sec
#train timer: Number of updates: 50. Total time: 837.81. Average time: 16.7563 sec
#validation timer: Number of updates: 50. Total time: 100.62. Average time: 2.0124 sec
#input timer: Number of updates: 50. Total time: 1519.39. Average time: 30.3877 sec
#eval timer: Number of updates: 50. Total time: 0.00. Average time: 0.0000 sec
#append timer: Number of updates: 50. Total time: 0.00. Average time: 0.0000 sec

#print("W[5,10] ", reg.reluLayer.W.get_value()[5,10])
#load(reg)
#reg.reluLayer.W.get_value()[5,10]
#w = reg.reluLayer.W.get_value()
#w[5,10] = 1
#reg.reluLayer.W.set_value(w)
#reg.reluLayer.W.get_value()[5,10]
#save(reg, "model_rootacc0.7469.save")
#datetime.now().strftime('%d-%m %H:%M')

if __name__ == "__main__":
    #from theano.tensor.shared_randomstreams import RandomStreams
    #os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')
    os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')
    #enron_dir = "F:/AIProjectsData/TREC2010/data/corpora/trec"
    
    #train_trees = get_nltk_parsed_trees_from_file('trees/train.txt')
    #valid_trees = get_nltk_parsed_trees_from_file('trees/dev.txt')
    #test_trees = get_nltk_parsed_trees_from_file('trees/test.txt')
    
    train_trees = load_trees.get_trees('trees/train.txt')
    valid_trees = load_trees.get_trees('trees/dev.txt')
    test_trees = load_trees.get_trees('trees/test.txt')
    
    
    nx = 50
    Evaluator.set_size(nx)
    LT = get_word_embeddings("../code/glove/glove.6B.{}d.txt".format(nx), RandomState(1234))
    len(LT)
    
    LT[UNKNOWN_WORD]
    LT["the"]
    
    initializeTrees(train_trees, LT)
    initializeTrees(valid_trees, LT)
    initializeTrees(test_trees, LT)
    
    print('... building the model')
    
    learning_rate=0.01
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=1000
    batch_size=40
    retain_probability = 0.8
    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_trees) // batch_size
    valid_batch_size = len(valid_trees)
    n_valid_batches = len(valid_trees) // valid_batch_size
    n_test_batches = len(test_trees) // batch_size    
    
    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  
    y = T.matrix('y')  
    z = T.matrix('z')    #for dropout
    
    rng = RandomState(1234)
    #srng = RandomStreams(1234)
    
    # Define RNN
    reg = rnn.RNN(rng=rng, X=x, Z=z, n_in=2*(Evaluator.SIZE+Evaluator.HIDDEN_SIZE), 
                  n_hidden=Evaluator.HIDDEN_SIZE, n_out=Evaluator.RES_SIZE
                  )
    
    #from theano.tensor import printing
    #printing.debugprint(reg.reluLayer.output)
    #
    #minibatch_index=0
    #trees = train_trees[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    #evaluator = Evaluator(reg)
    #(roots, x_val, y_val) = getInputArrays(reg, trees, evaluator)
    #z_val = rng.binomial(n=1, size=(x_val.shape[0], Evaluator.HIDDEN_SIZE), p=0.8)
    #o = reg.reluLayer.output.eval({x:x_val, z:z_val})
    #o.shape
    #o2 = train_model(x_val,y_val,z_val)
    
    
    cost = reg.cost(y) + L1_reg * reg.L1 + L2_reg * reg.L2_sqr
    
        
    validate_model = theano.function(
        inputs=[x,y,z],
        outputs=reg.errors(y)
    )
    
    gparams = [T.grad(cost, param) for param in reg.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(reg.params, gparams)
    ]
    
    cost_model = theano.function(
        inputs=[x,y,z],
        outputs=cost
    )
    
    train_model = theano.function(
        inputs=[x,y,z],
        outputs=cost,
        updates=updates
    )
    
    #train_times = 1
    run(n_epochs=1)


    

