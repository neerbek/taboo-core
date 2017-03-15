# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:20:15 2016

@author: neerbek
"""

import unittest

#Test disabled
print("note keras testing is disabled")
if False:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD
    
import numpy as np

data_dim = 20

nb_classes = 4
class KerasTest(unittest.TestCase):
    
    @unittest.skip
    def test_keras_train(self):
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, input_dim=data_dim, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(64, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, init='uniform'))
        model.add(Activation('softmax'))
        
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        
        # generate dummy training data
        x_train = np.random.random((1000, data_dim))
        y_train = np.random.random((1000, nb_classes))
        
        # generate dummy test data
        x_test = np.random.random((100, data_dim))
        y_test = np.random.random((100, nb_classes))
        
        model.fit(x_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True)
        score = model.evaluate(x_test, y_test, batch_size=16)
        print(score)
        
        model2 = Sequential()
        model2.add(Dense(64, input_dim=data_dim, init='uniform'))
        model2.add(Activation('tanh'))
        model2.add(Dropout(0.5))
        model2.add(Dense(256, init='uniform'))
        model2.add(Activation('tanh'))
        model2.add(Dropout(0.5))
        model2.add(Dense(256, init='uniform'))
        model2.add(Activation('tanh'))
        model2.add(Dropout(0.5))
        model2.add(Dense(nb_classes, init='uniform'))
        model2.add(Activation('softmax'))
        
        model2.compile(loss='categorical_crossentropy', optimizer=sgd)
        
        model2.fit(x_train, y_train, nb_epoch=200, batch_size=16, show_accuracy=True)
        score = model2.evaluate(x_test, y_test, batch_size=16)
        print(score)

if __name__ == "__main__":
    #import os
    #os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')
    unittest.main()
