# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:49:15 2017

@author: neerbek
"""

# start spyder with: OMP_NUM_THREADS=3 spyder3
import io
import sys
#import time
import time

import numpy
import theano
#import theano.tensor as T

import similarity.load_trees as load_trees
from ai_util import Timer
from StatisticTextParser import StatisticTextParser

MAX_SENTENCE_LENGTH=600      #set lower if you need parsing to happen faster
DEBUG_PRINT_VERBOSE=False
DEBUG_PRINT = True
DEBUG_PRINT_ON_SPLIT = False


class Timers:  #for parsing text
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

class RNNTimers: #timers for evaluating RNN
    def init(): #resetting timers
        RNNTimers.evaltimer2 = Timer("eval timer inner")
        RNNTimers.evaltimer = Timer("eval timer")
        RNNTimers.appendtimer = Timer("append timer")
        RNNTimers.getinputarraytimer = Timer("getInputArrays")
        RNNTimers.looptimer = Timer("loop")

RNNTimers.init()

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
        self.sentences = 0          #number of sentences 
        self.sentencesToBeSplit = 0 #number of sentences that have been split at least once
        self.splits = 0             #total number of splits
        self.emptySubTrees = 0      #empty (splited) sentences
        self.emptySentenceTrees = 0 #empty complete sentence
        self.failedSameSentences = 0  #sentence after parsing,deserialization and serialization was changed

def get_nltk_parsed_tree_from_sentence(l2, parser, timers, parserStatistics):
    fn = "get_nltk_parsed_tree_from_sentence"   #function name
    parserStatistics.sentences += 1
    l2 = load_trees.escape_sentence(l2)
    w = parser.tokenizer.tokenize(l2)
    #we generate a copy of l2 to test the final string from the parser (they should be the same)
    l2_copy = ""
    i = 0
    if i<len(w):
        l2_copy = w[0]
        i += 1
        while i<len(w):
            add_space=True
            e = w[i]
            if len(e)>1:
                pre = e[0:2]
                if pre=='.\n' or pre==',\n' or pre==':\n' or pre==';\n':
                    add_space=False
            elif e=='.' or e==',' or e==';' or e==':':
                add_space=False
            if add_space:
                l2_copy += " "            
            l2_copy += e        
            i += 1
    l2 = l2.strip()
    trees=[]
    if len(l2)>MAX_SENTENCE_LENGTH:
        parserStatistics.sentencesToBeSplit += 1
        if DEBUG_PRINT_VERBOSE:
            print("Going to split long sentence: {}".format(len(l2)))
            print(l2)
    subsentence_skipped=False
    while len(l2)>0:
        tmp = None
        if len(l2)>MAX_SENTENCE_LENGTH:
            index = l2.find(' ', int(MAX_SENTENCE_LENGTH/2))
            if index!=-1:
                parserStatistics.splits += 1
                if DEBUG_PRINT_ON_SPLIT:
                    print("Splitting long sentence: {}".format(len(l2)))
                tmp = l2[:index]
                l2 = l2[index+1:]
        if tmp is None:
            tmp = l2
            l2 = ""
        if DEBUG_PRINT_VERBOSE:
            print("subsentence is: " + tmp)
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
            print("Warning: " + fn + " marshall and unmarshalling differs" + "\n" + l2_copy + "\n" + l3)
            parserStatistics.failedSameSentences += 1
            root = None
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
    def getRatio(self):
        if self.word_count==0:
            return 0.0
        return (self.unknown_count + 0.0)/self.word_count
        
UNKNOWN_WORD = "*unknown*"
def updateWordVectors(node, lookup_table, nodeCounter):
    # assumes is_binary and has_only_words_at_leafs
    nodeCounter.incNode()
    if node.is_leaf():
        nodeCounter.incWord()
        word = node.word.lower()
        #word = node.word
        if word[-1]==',' and len(word)>1:
            word = word[:-1]  #remove trailing ,
        if word == "-lrb-":
            word = "("
        elif word == "-rrb-":
            word = ")"
        elif word == "-lsb-":
            word = "("
        elif word == "-rsb-":
            word = ")"
        elif word == "-lcb-":
            word = "("
        elif word == "-rcb-":
            word = ")"
        elif word == "-amp-":
            word = "&"

        if word in lookup_table:
            node.representation = lookup_table[word]
        else:
            nodeCounter.incUnknown()
            node.representation = lookup_table[UNKNOWN_WORD]
            #if nodeCounter.unknown_count<100:
            #    print("unknown word: \"" + word + "\"")
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
    if DEBUG_PRINT:
        print("Done with tree. Saw {} nodes, {} words and {} unknowns. Unknown ratio is {}".format(totalCounter.node_count, totalCounter.word_count, totalCounter.unknown_count, totalCounter.getRatio()))
    print("Done with tree. Saw {} nodes, {} words and {} unknowns. Unknown ratio is {}".format(totalCounter.node_count, totalCounter.word_count, totalCounter.unknown_count, totalCounter.getRatio()))

class Evaluator:
    RES_SIZE=5
    SIZE=None     #initialize this
    HIDDEN_SIZE=None
    hidden_rep = None
    leaf_rep = None
    def set_size(SIZE, HIDDEN_SIZE):
        Evaluator.SIZE = SIZE
        Evaluator.leaf_rep = numpy.zeros(Evaluator.SIZE)
        Evaluator.HIDDEN_SIZE = HIDDEN_SIZE
        Evaluator.hidden_rep = numpy.zeros(Evaluator.HIDDEN_SIZE)
    def __init__(self, reg):
        self.W = reg.reluLayer.W.eval()
        self.b = reg.reluLayer.b.eval()
        self.in_rep = None
        #self.in_rep = numpy.random.uniform(-1, 1, size=100)
        self.reg=reg
#        self.rep_generator = theano.function(
#            inputs=[reg.reluLayer.X],
#            outputs=[reg.reluLayer.output_no_dropout]
#        )
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
            #self.in_rep = numpy.concatenate(l)   #.reshape(1, 2*Evaluator.SIZE)
            #for ll in l:
            #    print(ll.shape)
            self.in_rep = numpy.concatenate(l)
            #self.in_rep = numpy.array(self.in_rep, copy=False, ndmin=2)
            #self.in_rep = self.in_rep.reshape(1, 2*(Evaluator.SIZE+Evaluator.HIDDEN_SIZE))
        except Exception:
            print("bad concatenation")
            raise
        #RNNTimers.evaltimer2.begin()
        lin = numpy.dot(self.in_rep, self.W) + self.b        
        lin = numpy.maximum(lin, 0, lin)
        #lin = self.reg.reluLayer.rep_generator(self.in_rep)
        #print(len(lin[0][0]))
        #lin = lin[0][0][0:Evaluator.HIDDEN_SIZE]
        #lin = numpy.array(lin)  #, copy=False)
        #lin = numpy.array(lin, copy=False)
        #lin = lin.reshape(Evaluator.HIDDEN_SIZE)
        #lin = lin.flatten()
        #lin = numpy.transpose(lin)
        #RNNTimers.evaltimer2.end()
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

#TODO: move to test
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
            

def addNodeRepresentations(reg, node, x_val, y_val, evaluator):
    if node == None:
        return
    if node.number_of_children()==0:
        return
    if node.number_of_children()==1:
        raise Exception("tree is not valid!")
        
        
    addNodeRepresentations(reg, node.left, x_val, y_val, evaluator)
    addNodeRepresentations(reg, node.right, x_val, y_val, evaluator)
    
    #RNNTimers.evaltimer.begin()
    node.representation = evaluator.get_representation(node.left, node.right)
    #RNNTimers.evaltimer.end()
    
    #RNNTimers.appendtimer.begin()
    y_val.append(node.label)
    x_val.append(evaluator.in_rep) 
    # we need to reshape in_rep but this will add 150% to running time
    #RNNTimers.appendtimer.end()

def getInputArrays(reg, trees, evaluator):
    """Generates input representations for use with the rnn in training and in evaluation. I.e. 
    formats the input which encoded as trees into x_val and y_val flat list"""
    #RNNTimers.getinputarraytimer.begin()
    list_x = []
    list_y = []
    list_root_indexes = []
    #RNNTimers.looptimer.begin()
    for t in trees:
        if t is None:
            raise Exception("Received a none tree")
        if t.left==None or t.right==None:
            raise Exception("one word tree")
        addNodeRepresentations(reg, t, list_x, list_y, evaluator)
        list_root_indexes.append(len(list_x)-1) #root is added last
    #RNNTimers.looptimer.end()
    #train_set_x = theano.shared(numpy.asarray(train_set_x), borrow = True)
    #train_set_y = theano.shared(numpy.asarray(train_set_y), borrow = True)        
    #list_x = [e.reshape(-1) for e in list_x]
    x_val = numpy.concatenate(list_x,axis=0).reshape(-1,2*(Evaluator.SIZE + Evaluator.HIDDEN_SIZE))
    #x_val = numpy.vstack(list_x)  - shower by 1.1 sec
    y_val = numpy.concatenate(list_y,axis=0).reshape(-1,Evaluator.RES_SIZE)
    #x_val = numpy.asarray(list_x)
    #y_val = numpy.asarray(list_y)
    x_val = x_val.astype(dtype=theano.config.floatX)
    y_val = y_val.astype(dtype=theano.config.floatX)
    if x_val.shape != (len(list_x), 2*(Evaluator.SIZE + Evaluator.HIDDEN_SIZE)):
        raise Exception("error in numpy conversion of x, shape was {}".format(x_val.shape))
    if y_val.shape != (len(list_y), Evaluator.RES_SIZE):
        raise Exception("error in numpy conversion of y")
    #RNNTimers.getinputarraytimer.end()
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
    res = (zeros + 0.0) / len(y_val)
    #print("rnn_enron: Total count {}, sensitive {} fraction sensitive {:.4f}".format(len(y_val), zeros, res))
    return res
