# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 08:16:27 2017

@author: neerbek
"""
import os
import time
import ast
import operator as op
import zipfile
from numpy.random import RandomState  # type: ignore
from typing import List


class Timer:
    def __init__(self, name="Timer"):
        self.elapsed = 0
        self.start = None
        self.count = 0
        self.name = name

    def begin(self):  # function cannot be named start
        self.start = time.time()

    def end(self, count=1):
        if self.start != None:
            end = time.time()
            self.elapsed += end - self.start
            self.count += count
            self.start = None
        else:
            print(
                "{} end() called, but timer is not started".format(self.name))
        return self

    def ratio(self):
        "seconds per count"
        if self.count == 0:
            return 0
        return (self.elapsed + 0.0) / self.count

    def report(self, count=None):
        if count != None:
            self.count = count
        print(
            "{}: Number of updates: {}. Total time: {:.2f}. Average time: {:.4f} sec".
            format(self.name, self.count, self.elapsed, self.ratio()))


class TimerList:
    def __init__(self):
        self.timers = []  # type: List[Timer]
        self.lastReport = time.time()
        self.totalTimer = None  # type:Timer
        self.forwardTimer = None  # type:Timer
        self.backwardTimer = None  # type:Timer

    def addTimer(self, timer):
        self.timers.append(timer)

    def end(self):
        for t in self.timers:
            if t.start != None:
                t.end()

    def update(self):
        end = time.time()
        for t in self.timers:
            if t.start != None:
                t.elapsed += end - t.start
                t.start = end

    def do_report(self, min_seconds=-1, t=time.time()):
        if (t - self.lastReport < min_seconds):
            return False
        return True

    def report(self, min_seconds=-1, update_timers=False):
        t = time.time()
        if not self.do_report(min_seconds, t):
            return False
        if update_timers:
            self.update()
        self.lastReport = t
        for timer in self.timers:
            timer.report()
        return True


# evaluate math expressions, from
# https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
#
# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}


def eval_expr(expr):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        # type: ignore
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))    # type: ignore
    else:
        raise TypeError(node)


class AIFileWrapper():
    def __init__(self, filename):
        self.filename = filename
        self.myzip = None
        self.fd = None
        self.is_zip = None
        _ = self.getFilename()  # initialize is_zip

    def __enter__(self):
        zipfilename, internalfilename = self.getFilename()
        if zipfilename is not None:
            self.myzip = zipfile.ZipFile(zipfilename)
            # always binary, even with mode='r' [sic]
            self.fd = self.myzip.open(internalfilename, mode='r')
        else:
            # io.open does not support binary it seems
            self.fd = open(self.filename, 'rb')
        return self

    def __exit__(self, type, value, traceback):
        # Exception handling here
        if self.fd != None:
            self.fd.close()
        if self.myzip != None:
            self.myzip.close()

    def toStrippedString(self, line, encoding="utf8"):
        # print(line)
        line = str(line, encoding=encoding)
        if line.endswith("\r\n"):
            line = line[:-2]
        if line.endswith("\n"):
            line = line[:-1]
        # Strips newline. Consider:
        # http://stackoverflow.com/questions/509446/python-reading-lines-w-o-n
        return line

    def getFilename(self):
        self.is_zip = False
        index = self.filename.find("$")
        zipfilename = None
        internalfilename = self.filename
        if index != -1:  # assume zipfile
            self.is_zip = True
            zipfilename = self.filename[:index]
            internalfilename = self.filename[index + 1:]
        return (zipfilename, internalfilename)

    def exists(self):
        if self.myzip != None:
            return True
        zipfilename, internalfilename = self.getFilename()
        if zipfilename is None:
            return os.path.exists(internalfilename)
        return os.path.exists(zipfilename)

    def readAll(self):
        return self.fd.read()


def shuffleList(a, rng=RandomState(1234)):
    """Expects a to be a list type. Shuffles all elements and return new list"""
    perm = rng.permutation(
        len(a))  # we have seen issues using the built-in shuffle
    aNew = [a[i] for i in perm]
    return aNew
