# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:58:46 2017

@author: neerbek
"""

from flask import Flask
from flask import request
from flask import Response
import json

app = Flask(__name__)

import numpy as np
import sys

import rnn_enron
import server_rnn
import server_rnn_helper
import inference_enron

import server_enron_helper

import similarity.load_trees as load_trees
       
serverState = server_enron_helper.ServerState()
keywordState = server_enron_helper.KeywordState()

if __name__ == "__main__":
    serverState.initialize()  #calls server_rnn.initialize_state
    t = server_rnn.Trainer()
    server_rnn.load_model(t)
    keywordState.initialize(server_rnn.state)

def is_none(l):
    for e in l:
        if e==None or len(e)==0:
            return True
    return False
def is_positive(l):
    for e in l:
        if e<=0:
            return False
    return True


#for checking connection
#http://localhost:5000/hello_world
@app.route("/hello_world")
def hello_world():
    pres = []
    pres.append(server_enron_helper.Prediction(42,87, prediction=1))
    p = server_enron_helper.Predictions(pres)

    return response_success(p)

#http://localhost:5000/run_model/rnn?file=text
@app.route("/run_model/rnn")
def run_rnn():
    print('run_rnn')
    text = request.args.get('text')
    if text is None or len(text)==0:
        return response_error("No text received")
    else: 
        print("parsing text")
        indexed_sentences = server_rnn_helper.get_indexed_sentences(text)
        trees = server_rnn_helper.get_nltk_trees(0, indexed_sentences)
        for t in trees:
            t.replace_nodenames("0")
        rnn_enron.initializeTrees(trees, server_rnn.state.LT)
        print("getting predictions")
        server_rnn.get_predictions(serverState.rnn, indexed_sentences)
        for s in indexed_sentences:
            s.pred = (s.pred==4)   #is_sensitive
        res = server_enron_helper.index_sentences_to_predictions(indexed_sentences)
        return response_success(res)

        
@app.route("/run_model/keyword")
def run_keyword():
    print('run_keyword')
    text = request.args.get('text')
    if text is None or len(text)==0:
        return response_error("No text received")
    else:
        print("getting predictions")
        indexed_sentences = inference_enron.get_predictions(text, keywordState.indicators)
        res = server_enron_helper.index_sentences_to_predictions(indexed_sentences)
        return response_success(res)


@app.route("/train_data")
def load_model_data():
    print('train_data loading')
    labelfile = request.args.get('labelfile')
    data_dir = request.args.get('data_dir')
    data_label = request.args.get('data_label')
    total_ratio_str = request.args.get('total_ratio')
    train_ratio_str = request.args.get('train_ratio')
    dev_ratio_str = request.args.get('dev_ratio')
    if is_none([data_dir, labelfile, data_label, total_ratio_str, train_ratio_str, dev_ratio_str]):
        return response_error("Load data needs parameters labelfile, data_dir, data_label, total_ratio, train_ratio, dev_ratio")
    total_ratio = float(total_ratio_str)
    train_ratio = float(train_ratio_str)
    dev_ratio = float(dev_ratio_str)
    if not is_positive([total_ratio, train_ratio, dev_ratio]):
        return response_error("Load data needs parameters total_ratio, train_ratio, dev_ratio to be postive numbers (below 1)")
#    doc2 = load_labeled_documents("enron_labels.txt","/home/neerbek/Dropbox/DLP/trec/legal10", "201")
    doc2 = server_enron_helper.load_labeled_documents(labelfile,data_dir, data_label)
    print("***totalratio:", len(doc2), total_ratio, train_ratio, dev_ratio)
    total_index = int(len(doc2)*total_ratio)
    train_index = int(total_index*train_ratio)
    dev_index = int(len(doc2)*dev_ratio)
    print("***totalindex:", total_index, train_index, dev_index)
    train_trees = server_rnn_helper.get_trees(doc2[:train_index], server_rnn.state.LT)
    dev_trees = server_rnn_helper.get_trees(doc2[train_index:dev_index], server_rnn.state.LT)
    test_trees = server_rnn_helper.get_trees(doc2[dev_index:], server_rnn.state.LT)
    print("***trees loaded")
    serverState.train_trees = train_trees
    serverState.valid_trees = dev_trees
    serverState.test_trees = test_trees
    print('train_data DONE loading')
    sys.stdout.flush()    
    return response_success(ResponseString("load completed"))

@app.route("/train_data_from_file")
def load_model_data_from_file():
    print('train_data from file loading')
    data_file = request.args.get('data_file')
    total_ratio_str = request.args.get('total_ratio')
    train_ratio_str = request.args.get('train_ratio')
    dev_ratio_str = request.args.get('dev_ratio')
    if is_none([data_file, total_ratio_str, train_ratio_str, dev_ratio_str]):
        return response_error("Load data needs parameters data_file, total_ratio, train_ratio, dev_ratio")
    total_ratio = float(total_ratio_str)
    train_ratio = float(train_ratio_str)
    dev_ratio = float(dev_ratio_str)
    if not is_positive([total_ratio, train_ratio, dev_ratio]):
        return response_error("Load data needs parameters total_ratio, train_ratio, dev_ratio to be postive numbers (below 1)")
    doc2 = load_trees.get_trees(data_file)
    print("***totalratio:", len(doc2), total_ratio, train_ratio, dev_ratio)
    total_index = int(len(doc2)*total_ratio)
    train_index = int(total_index*train_ratio)
    dev_index = int(len(doc2)*dev_ratio)
    print("***totalindex:", total_index, train_index, dev_index)
    train_trees = doc2[:train_index]
    dev_trees = doc2[train_index:dev_index]
    test_trees = doc2[dev_index:]
    print("***trees loaded")
    serverState.train_trees = train_trees
    serverState.valid_trees = dev_trees
    serverState.test_trees = test_trees
    print('train_data from file DONE loading')
    sys.stdout.flush()
    return response_success(ResponseString("load completed"))

def train_rnn_impl(trainer, n_epochs):
    trainer.train(serverState.rnn, n_epochs, serverState.rng)

@app.route("/train_model/rnn")
def train_rnn():
    print('train_rnn')
    print("loading model")
    trainer = server_rnn.Trainer()

    learning_rate = request.args.get('learning_rate')
    L1_reg = request.args.get('L1_reg')
    L2_reg = request.args.get('L2_reg')
    n_epochs = request.args.get('n_epochs')
    batch_size = request.args.get('batch_size')
    retain_probability = request.args.get('retain_probability')
    if is_none([learning_rate, L1_reg, L2_reg, n_epochs, batch_size, retain_probability]):
        return response_error("Training needs parameters learning_rate, L1_reg, L2_reg, n_epochs, batch_size, retain_probability")
    trainer.retain_probability = float(retain_probability)
    trainer.batch_size = int(batch_size)
    trainer.L2_reg = float(L2_reg)
    trainer.L1_reg = float(L1_reg)
    trainer.learning_rate =float(learning_rate)
    n_epochs = int(n_epochs)
    trainer.update_batch_size()
    if not is_positive([trainer.learning_rate, n_epochs, trainer.batch_size]):
        return response_error("Training needs parameters learning_rate, n_epochs, batch_size to be above zero")
    train_rnn_impl(trainer, n_epochs)
    print("train RNN DONE")
    sys.stdout.flush()
    return response_success(ResponseString("training completed"))

@app.route("/train_model/keyword")
def train_keyword():
    print("train_keyword")
    cut_off = request.args.get('cut_off')
    if is_none([cut_off]):
        return response_error("Training keywords needs cutoff")
    cut_off = float(cut_off)
    keywordState.initialize(server_rnn.state, cut_off)
    print("train_keyword DONE")
    sys.stdout.flush()
    return response_success(ResponseString("keyword training completed"))

@app.route("/examples")
def get_examples():
    train = server_rnn.state.train_trees
    rng = serverState.rng
    train = rng.choice(train, 50, replace=False)
    res = []
    for s in train:
        y_target = np.argmax(s.label)
        #print("s.label", s.label)
        txt = load_trees.output_sentence(s)
        txt = load_trees.unescape_sentence(txt) + "\n"
        if len(txt)<22 or len(txt)>140:
            continue #ignore this tree
        sent = server_rnn_helper.IndexSentence(0, txt)
        sent.y_target = y_target
        sent.endIndex = len(txt)
        res.append(sent)
    res = res[:10]
    trees = server_rnn_helper.get_nltk_trees(0, res)
    for t in trees:
        t.replace_nodenames("0")
    rnn_enron.initializeTrees(trees, server_rnn.state.LT)
    server_rnn.get_predictions(serverState.rnn, res)
    indexed_sentences = []
    count = 0
    for s in res:
        if len(indexed_sentences)==5:
            break
        if s.y_target == s.pred:
            sent = server_rnn_helper.IndexSentence(0, s.sentence)
            print("s.pred", s.pred)
            sent.pred = (s.pred==4)   #is_sensitive
            sent.endIndex = s.endIndex
            indexed_sentences.append(sent)
        count += 1
    print("visited:", count)
    for s in indexed_sentences:
        print(s.pred, len(s.sentence), s.sentence)
    sys.stdout.flush()
    l = server_enron_helper.index_sentences_to_predictions(indexed_sentences)
    return response_success(l)

class ResponseString:
    def __init__(self, msg):
        self.msg = msg
        
def response_error(msg):
    res = Response(response=msg,
                        status=500,
                        mimetype="text/plain")
    print(msg)
    return res

def toJSON(e):
        return json.dumps(e, default=lambda o: vars(o))
        
def response_success(value, root_tag = None):
    res = toJSON(value)
    print(value)
    if root_tag is not None:
        res = '{"' + root_tag +'": ' + res + '}'
    return res


if __name__ == "__main__":
   app.run()

