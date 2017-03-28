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
            self.start = None
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

class TimerList:
    def __init__(self):
        self.timers = []
        self.lastReport = time.time()
    def addTimer(self, timer):
        self.timers.append(timer)
    
    def end(self):
        for t in self.timers:
            if t.start!=None:
                t.end()
    def update(self):
        end = time.time()
        for t in self.timers:
            if t.start!=None:
                t.elapsed += end-t.start
                t.start = end
    def do_report(self, min_seconds=-1, t = time.time()):
        if (t-self.lastReport <min_seconds):
            return False
        return True
    def report(self, min_seconds=-1, update_timers=False):
        t = time.time()
        if not self.do_report(min_seconds, t):
            return False
        if update_timers:
            self.update()
        self.lastReport = t
        for t in self.timers:
            t.report()
        return True
