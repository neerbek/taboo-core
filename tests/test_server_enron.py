# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:08:31 2017

@author: neerbek
"""

import unittest
import os
from flask import Response
import json

class Args:
    def __init__(self):
        self.args = {}
        
    def get(self, key):
        if key in self.args:
            return self.args[key]
        return None
    
class Request:
    def __init__(self):
        self.args = Args()

example_text = """Please have a look at enclosed worksheets. 
As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact. This may impact the final decision.  
We may have to move to cash margining if necessary."""

example_text2 = """Please have a look at enclosed worksheets. 
As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact. This may impact the final decision.  
We may have to move to cash margining if necessary. prepay model"""


import server_enron
import server_enron_helper
import server_rnn
import rnn_enron

import RunTimer

DEBUG_PRINT = False
rnn_enron.MAX_SENTENCE_LENGTH=80  #approx 14secs, if =160 approx 30secs
rnn_enron.DEBUG_PRINT_VERBOSE = DEBUG_PRINT
rnn_enron.DEBUG_PRINT = DEBUG_PRINT

state = {}

def initialize_model(num_wordvectors=5000, num_trees=1000):
    if len(state)==0:
        server_rnn_state = server_enron.serverState.server_rnn_state
        state['LT'] = server_rnn_state.LT   #hack - we know serverState has been loaded the first time we are called
        trainer = server_rnn.Trainer()
        server_rnn_state.load_trees(trainer)
        state['trt'] = server_rnn_state.train_trees
        state['vat'] = server_rnn_state.valid_trees
        state['tet'] = server_rnn_state.test_trees

    server_enron.serverState = server_enron_helper.ServerState(max_embedding_count=1)
    server_rnn_state = server_enron.serverState.server_rnn_state
    
    server_rnn_state.LT = state['LT']
    server_enron.keywordState = server_enron_helper.KeywordState()
    server_enron.serverState.rnn = server_rnn.RNNWrapper()
    
    server_rnn_state.train_trees = state['trt'][:num_trees]
    server_rnn_state.valid_trees = state['vat'][:num_trees]
    server_rnn_state.test_trees = state['tet'][:num_trees]
    trainer = server_rnn.Trainer()
    server_rnn_state.init_trees(trainer)        

    server_enron.serverState.load_model()
    server_enron.keywordState.initialize(server_enron.serverState.server_rnn_state)


class ServiceTest(unittest.TestCase):
    def setUp(self):
        self.timer = RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)


    def test_run_rnn1(self):
        initialize_model()
        server_enron.request = Request()
        res = server_enron.run_rnn()
        if not isinstance(res, Response):
            self.assertTrue(False,"expected Response class")
        print(res.status)
        self.assertTrue(res.status_code==500)


    def test_run_rnn2(self):
        initialize_model()
        server_enron.request = Request()
        server_enron.request.args.args['text'] = example_text        
        res = server_enron.run_rnn()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        self.assertTrue(d['predictions']!=None)

    def test_run_rnn3(self):
        initialize_model(num_wordvectors=10000000, num_trees=11000)
        server_enron.request = Request()
        server_enron.request.args.args['text'] = "Sentence fragments are representative of their quoted statement"
        res = server_enron.run_rnn()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        print(res)
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        print(d['predictions'])
        self.assertEqual(1, d['predictions'][0]['prediction'], "expected first result to be sensitive")
    
    def test_run_keyword1(self):
        initialize_model()
        server_enron.request = Request()
        res = server_enron.run_keyword()
        if not isinstance(res, Response):
            self.assertTrue(False,"expected Response class")
        print(res.status)
        self.assertTrue(res.status_code==500)

    
    def test_run_keyword2(self):
        initialize_model()
        server_enron.request = Request()
        server_enron.request.args.args['text'] = example_text
        res = server_enron.run_keyword()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        self.assertTrue(d['predictions']!=None)
        self.assertTrue(len(d['predictions'])==1)

    
    def test_run_keyword3(self):
        initialize_model()
        server_enron.request = Request()
        server_enron.request.args.args['text'] = example_text2
        res = server_enron.run_keyword()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        self.assertTrue(d['predictions']!=None)
        self.assertEqual(1, len(d['predictions']), "expected a prediction")

    def test_load_keyword1(self):
        initialize_model()
        server_enron.request = Request()
        server_enron.request.args.args['cut_off'] = '0.9'
        res = server_enron.train_keyword()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        self.assertTrue(d['msg']!=None)

    
    def test_load_model(self):
        initialize_model()
        req = Request()
        server_enron.request = req
        res = server_enron.load_model_data()
        if not isinstance(res, Response):
            self.assertTrue(False,"expected Response class")
        self.assertTrue(res.status_code==500)

    def test_load_model2(self):
        initialize_model()
        req = Request() 
        req.args.args['labelfile'] = 'tests/resources/enron_labels_test.txt'
        req.args.args['data_dir'] = os.path.join(os.getcwd(), 'tests/resources/enron_shortdata_test')   #'/home/neerbek/Dropbox/DLP/trec/legal10'
        req.args.args['data_label'] = '201'
        req.args.args['total_ratio'] = '1'
        req.args.args['train_ratio'] = '0.6'
        req.args.args['dev_ratio'] = '0.75'
        server_enron.request = req
        res = server_enron.load_model_data()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(d['msg']!=None)
        self.assertTrue(len(server_enron.serverState.train_trees)==11)
        #    rand = RandomState(374637)
        #    is_positive = rand.randint(10, size=len(doc2))
        #    #for debugging
        #    for i in range(len(doc2)):
        #        d = doc2[i]
        #        is_p = is_positive[i]
        #        if is_p>6:
        #            d.enron_label.relevance="4"

    
    def test_load_model3(self):
        initialize_model()
        req = Request() 
        req.args.args['data_file'] = os.path.join(os.getcwd(), 'trees/train.txt')
        req.args.args['max_count'] = '1000'
        req.args.args['total_ratio'] = '1'
        req.args.args['train_ratio'] = '0.6'
        req.args.args['dev_ratio'] = '0.75'
        server_enron.request = req
        res = server_enron.load_model_data_from_file()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(d['msg']!=None)
        self.assertEqual(600, len(server_enron.serverState.train_trees), "number of new training trees does not match")

    
    def test_train_rnn1(self):
        initialize_model()
        server_enron.request = Request()
        trainer = server_rnn.Trainer()
        server_enron.serverState.server_rnn_state.load_trees(trainer, max_tree_count=100)
        server_enron.train_rnn_impl(trainer,1)

    
    def test_train_rnn2(self):
        initialize_model()
        server_enron.request = Request()
        trainer = server_rnn.Trainer()
        server_enron.serverState.server_rnn_state.load_trees(trainer, max_tree_count=100)
        req = Request()
        server_enron.request = req
        res = server_enron.train_rnn()
        if not isinstance(res, Response):
            self.assertTrue(False,"expected Response class")
        self.assertTrue(res.status_code==500)


    def test_train_rnn3(self):
        initialize_model()
        server_enron.request = Request()
        trainer = server_rnn.Trainer()
        server_enron.serverState.server_rnn_state.load_trees(trainer, max_tree_count=100)
        req = Request()
        req.args.args['learning_rate'] = '0.001'
        req.args.args['L1_reg'] = '0.001'
        req.args.args['L2_reg'] = '0.001'
        req.args.args['n_epochs'] = '1'
        req.args.args['batch_size'] = '50'
        req.args.args['retain_probability'] = '0.8'
        server_enron.request = req
        res = server_enron.train_rnn()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        print(res)
        self.assertTrue(len(d)==1)

    def test_get_examples(self):
        initialize_model()
        server_enron.request = Request()
        res = server_enron.get_examples()        
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(len(d)==1)
        for p in d['predictions']:
            print(p['prediction'], p['text'])

if __name__ == "__main__":
    unittest.main()
