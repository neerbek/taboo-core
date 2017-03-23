# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 08:16:27 2017

@author: neerbek
"""
import time

class Timer:
    def __init__(self, name = "Timer"):
        self.elapsed = 0
        self.start = None
        self.count = 0
        self.name = name
    def begin(self):  #function cannot be named start
        self.start = time.time()
    def end(self, count = 1):
        if self.start!=None:
            end = time.time()
            self.elapsed += end -self.start
            self.count += count
        else:
            print("{} end() called, but timer is not started".format(self.name))
        return self
    def ratio(self):
        "seconds per count"
        if self.count == 0:
            return 0
        return (self.elapsed + 0.0) / self.count
    def report(self, count=None):
        if count!=None:
            self.count = count
        print("{}: Number of updates: {}. Total time: {:.2f}. Average time: {:.4f} sec".format(self.name, self.count, self.elapsed, self.ratio()))
