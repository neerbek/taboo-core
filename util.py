# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:33:24 2016

@author: neerbek
"""

from importlib import reload
import time
import subprocess
import io

import os
reload(os)  #to get rid of warning from reload being unused :)

def set_working_dir():
    import os
    print(os.getcwd())
    os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')
    os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python/embeddings')
    os.chdir('/home/neerbek/jan/phd/DLP/python')
    os.chdir('c:/Users/neerbek/jan/Macro')

def assert_true(val, msg=None):
    if not val:
        out = "Assertion failed"
        if not msg == None:
            out += " " + msg
        raise Exception(out)
        
def write_file(sentences, filename = "sentences.txt", maximum = -1, do_log=True):
    start = time.time()
    with io.open(filename,'w+',encoding='utf8') as f:
        for i, s in enumerate(sentences):
            f.write(s + "\n")
            if i==maximum:
                print("Maximum reached", i)
                break
            if do_log and i % 2000 == 0:
                print(i, "time elapsed is:", time.time() - start)
    done = time.time()
    if do_log:
        print("File written. Time elapsed is:", done - start)
    
def run_command(command):
    if not isinstance(command, list):
        raise Exception("command must be an array")        
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)  #note shell=True is potentially unsafe (security-wise)
    return iter(p.stdout.readline, b'')

def print_command(command):
    for i,l in enumerate(run_command(command)):
        print(l)
