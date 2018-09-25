# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:01:17 2017

@author: neerbek
"""
import os
import datetime

class Timer:
    def __init__(self):
        self.tick = datetime.datetime.now()

    def report(self, testInstance, testFilename):
        self.tock = datetime.datetime.now()
        diff = self.tock - self.tick
        print("Time used in test (" + os.path.basename(testFilename) + ")", testInstance.id().split('.')[-1], (diff.total_seconds()), "sec")
