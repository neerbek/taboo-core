# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 08:14:51 2017

@author: neerbek
"""

import os
os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')

import numpy
from numpy.random import RandomState

import ai_util
ai_util.reload(ai_util)
from ai_util import TrainTimer

def timeAppend(): 
    totaltimer = TrainTimer("totaltimer")
    totaltimer.begin()
    nx = 50
    ny=5
    sample = 50
    count = 1000000
    prng = RandomState(1234)
    rx_in = []
    ry_in = []
    for i in range(sample):
        rx_in.append(prng.uniform(-1, 1, size=nx))
        ry_in.append(prng.uniform(0, 1, size=ny))
    c = prng.randint(sample, size=count)
    empty = None
    x_in = [empty for i in range(count)]
    y_in = [empty for i in range(count)]
    for index in range(count):
        i = c[index]
        x_in[index] = rx_in[i]
        y_in[index] = ry_in[i]
    x_val=[empty for i in range(count)]
    y_val=[empty for i in range(count)]
    appendtimer = TrainTimer("appendtimer")
    appendtimer.begin()
    for i in range(count):
        #appendtimer.begin()
        y_val[i] = y_in[i]
        x_val[i] = x_in[i]
        #appendtimer.end()
    appendtimer.end()
    x2_val=[empty for i in range(count)]
    y2_val=[empty for i in range(count)]
    appendtimer2 = TrainTimer("appendtimer2")  
    appendtimer2.begin()
    for i in range(count):
        #appendtimer2.begin()
        y2_val[i] = y_in[i]
        x2_val[i] = x_in[i].reshape(1,nx)
        #appendtimer2.end()
    appendtimer2.end()
    x3_val=[empty for i in range(count)]
    y3_val=[empty for i in range(count)]
    appendtimer3 = TrainTimer("appendtimer3_copy")  
    appendtimer3.begin()
    for i in range(count):
        #appendtimer2.begin()
        y3_val[i] = numpy.copy(y_in[i])
        x3_val[i] = numpy.copy(x_in[i])
        #appendtimer2.end()
    appendtimer3.end()
    index = int(prng.uniform(0, count))
    print(y_val[index], x_val[index])
    print(y2_val[index], x2_val[index])
    print(y3_val[index], x3_val[index])
    totaltimer.end().report()
    appendtimer.report()
    appendtimer2.report()
    appendtimer3.report()

timeAppend()

#random for all vals in x_in/y_in
#totaltimer: Number of updates: 1. Total time: 5.19. Average time: 5.1873 sec
#appendtimer: Number of updates: 1000000. Total time: 0.36. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.79. Average time: 0.0000 sec

#random for sample
#totaltimer: Number of updates: 1. Total time: 4.50. Average time: 4.5045 sec
#appendtimer: Number of updates: 1000000. Total time: 0.47. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.76. Average time: 0.0000 sec

#random for sample, preallocate random lists
#totaltimer: Number of updates: 1. Total time: 4.24. Average time: 4.2408 sec
#appendtimer: Number of updates: 1000000. Total time: 0.36. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.74. Average time: 0.0000 sec

#random for sample, preallocate random lists, do not use numpy.copy
#totaltimer: Number of updates: 1. Total time: 2.11. Average time: 2.1148 sec
#appendtimer: Number of updates: 1000000. Total time: 0.36. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.68. Average time: 0.0000 sec

#random for sample, preallocate random lists, do not use numpy.copy, using empty value for preallocation
#totaltimer: Number of updates: 1. Total time: 2.08. Average time: 2.0813 sec
#appendtimer: Number of updates: 1000000. Total time: 0.35. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.67. Average time: 0.0000 sec

#random for sample, preallocate random lists for all lists, do not use numpy.copy, using empty value for preallocation
#totaltimer: Number of updates: 1. Total time: 2.03. Average time: 2.0340 sec
#appendtimer: Number of updates: 1000000. Total time: 0.30. Average time: 0.0000 sec
#appendtimer2: Number of updates: 1000000. Total time: 0.60. Average time: 0.0000 sec

#random for sample, preallocate random lists for all lists, do not use numpy.copy, using empty value for preallocation
#no inner timers
#totaltimer: Number of updates: 1. Total time: 0.74. Average time: 0.7384 sec
#appendtimer: Number of updates: 1. Total time: 0.08. Average time: 0.0770 sec
#appendtimer2: Number of updates: 1. Total time: 0.35. Average time: 0.3517 sec

#as before, with youtube running
#totaltimer: Number of updates: 1. Total time: 0.80. Average time: 0.7959 sec
#appendtimer: Number of updates: 1. Total time: 0.08. Average time: 0.0820 sec
#appendtimer2: Number of updates: 1. Total time: 0.38. Average time: 0.3764 sec

#as before, with youtube running and numpy.copy for arrays
#totaltimer: Number of updates: 1. Total time: 3.19. Average time: 3.1866 sec
#appendtimer: Number of updates: 1. Total time: 0.08. Average time: 0.0841 sec
#appendtimer2: Number of updates: 1. Total time: 0.38. Average time: 0.3789 sec
#appendtimer3_copy: Number of updates: 1. Total time: 2.32. Average time: 2.3229 sec


