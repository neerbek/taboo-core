# -*- coding: utf-8 -*-
"""
testing
"""

import unittest
import numpy as np
import time
import theano
import theano.tensor as T
from theano import function, config, shared
import datetime

class TheanoTest(unittest.TestCase):
    def setUp(self):
        self.tick = datetime.datetime.now()

    def tearDown(self):
        self.tock = datetime.datetime.now()
        diff = self.tock - self.tick
        print("Time used in test (TheanoTest)", self.id().split('.')[-1], (diff.total_seconds()), "sec")

    #modified copy of server_enron_helper.get_trees
    def test_dot_time(self):
        A = np.random.rand(800,1000).astype(theano.config.floatX)
        B = np.random.rand(1000,800).astype(theano.config.floatX)
        np_start = time.time()
        AB = A.dot(B)
        np_end = time.time()
        X,Y = theano.tensor.matrices('XY')
        mf = theano.function([X,Y],X.dot(Y))
        t_start = time.time()
        tAB = mf(A,B)
        t_end = time.time()
        print("NP time: %f[s], theano time: %f[s] (times should be close when run on CPU!)" %(
                                                   np_end-np_start, t_end-t_start))
        print("Result difference: %f" % (np.abs(AB-tAB).max(), ))

    def test_using_cpu_or_gpu(self):
        vlen = 30*768 #10 * 30 * 768 # 10 x #cores x # threads per core
        rng = np.random.RandomState(22)
        iters = vlen
        
        x = shared(np.asarray(rng.rand(vlen), config.floatX))
        f = function([], T.exp(x))
        print(f.maker.fgraph.toposort())
        t0 = time.time()
        for i in range(1000):
            r = f()
        t1 = time.time()
        print('Looping %d times took' % iters, t1 - t0, 'seconds')
        print('Result is', r)
        
        if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
            print('Used the cpu')
        else:
            print('Used the gpu')
            
    def test_function(self):
        X = theano.shared(value=np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]]), name='X')
        y = theano.shared(value=np.asarray([[1], [0], [1], [0]]), name='y')
        rng = np.random.RandomState(1234)
        LEARNING_RATE = 0.01
         
        def layer(n_in, n_out):
            return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=np.float64), name='W', borrow=True)
        #dtype=theano.config.floatX
         
        W1 = layer(2, 3)
        W2 = layer(3, 1)
         
        output = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(X, W1)), W2))
        cost = T.sum((y - output) ** 2)
        updates = [(W1, W1 - LEARNING_RATE * T.grad(cost, W1)), (W2, W2 - LEARNING_RATE * T.grad(cost, W2))]
         
        train = theano.function(inputs=[], outputs=[], updates=updates)
        test = theano.function(inputs=[], outputs=[output])
         
        for i in range(9999):
            if (i+1) % 10000 == 0:
                print(i+1)
            train()
        print(test()) 


if __name__ == "__main__":
    #import os
    #os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')
    unittest.main()

#2000x100000  5.83
#4000x100000 22.43 
#4000x200000 46.72
#4000x400000 (out of mem 32GB)
#8000x200000 (out of mem 32GB)
#8000x20000 8.13 vs. 24.97 