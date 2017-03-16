# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:08:31 2017

@author: neerbek
"""

import unittest
import os
from flask import Response
import json

#os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')
#os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')

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
import server_rnn
import rnn_enron

DEBUG_PRINT = False
rnn_enron.MAX_SENTENCE_LENGTH=80  #approx 14secs, if =160 approx 30secs
rnn_enron.DEBUG_PRINT_VERBOSE = DEBUG_PRINT
rnn_enron.DEBUG_PRINT = DEBUG_PRINT

def initialize_model(num_wordvectors=5000, num_trees=1000):
    server_enron.serverState.initialize(num_wordvectors)            
    if server_rnn.state.train_trees==None:
        t = server_rnn.Trainer()
        server_rnn.load_model(t, num_trees)
    server_enron.keywordState.initialize(server_rnn.state)


class ServiceTest(unittest.TestCase):

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
        self.assertTrue(d['predictions'][0]['prediction']==1)
    
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
        self.assertTrue(len(d['predictions'])==1)

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
        print("***jan was here")
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
        req.args.args['total_ratio'] = '1'
        req.args.args['train_ratio'] = '0.6'
        req.args.args['dev_ratio'] = '0.75'
        server_enron.request = req
        res = server_enron.load_model_data_from_file()
        if not isinstance(res, str):
            self.assertTrue(False,"expected string response")
        d = json.loads(res)
        self.assertTrue(d['msg']!=None)
        self.assertTrue(len(server_enron.serverState.train_trees)==5400)

    
    def test_train_rnn1(self):
        initialize_model()
        server_enron.request = Request()
        trainer = server_rnn.Trainer()
        server_rnn.load_model(trainer, max_count=100)
        server_enron.train_rnn_impl(trainer,1)

    
    def test_train_rnn2(self):
        initialize_model()
        server_enron.request = Request()
        trainer = server_rnn.Trainer()
        server_rnn.load_model(trainer, max_count=100)
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
        server_rnn.load_model(trainer, max_count=100)
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
