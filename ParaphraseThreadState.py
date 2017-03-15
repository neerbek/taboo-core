# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:29:56 2016

@author: neerbek
"""

from threading import Thread
from threading import Lock
import queue
import time

NO_BLOCK = False

class ParaphraseThreadState:
    def __init__(self):
        self.threads = {}
        self.threads_done = queue.Queue()
        self.work = queue.Queue()
        self.completed_work = queue.Queue()
        self.failed_work = queue.Queue()
        
    def add_thread(self, tname, thread):  #to be called from main before threads starting
        self.threads[tname] = thread
        
    def start_threads(self):
        for t in self.threads.values():
            t.start()
        
    def thread_done(self, thread_name):
        self.threads_done.put(thread_name)
        
    def is_all_threads_done(self):
        return self.threads_done.qsize()==len(self.threads)
        
    def put_work(self, w):
        return self.work.put(w)  # may block

    def get_work(self):
        return self.work.get(NO_BLOCK)  # may throw Queue.Empty
        
    def remaining_work(self):
        return self.work.qsize()
        
    def has_more_work(self):
        return not self.work.empty()

    def work_done(self, w):
        self.completed_work.put(w)

    def work_failed(self, w):
        self.failed_work.put(w)
        
        
class ThreadContainer:
    def __init__(self, name, state, myparent):
        self.name = name
        self.state = state
        self.t = None
        self.mutex = Lock()
        self.stop_signal = False
        self.the_log = []
        self.current_work = None
        self.myparent = myparent
        
    def log(self, msg):
        with self.mutex:
            self.the_log.append(msg)

    def get_log(self):
        res = []
        with self.mutex:
            res.extend(self.the_log)
        return res
        
    def do_stop(self):
        with self.mutex:
            self.stop_signal = True

    def is_stop(self):
        with self.mutex:
            return self.stop_signal
        
    def loop_work(self, work):          #may be extended by subclasses
        self.myparent.run(work)

    def get_work(self):          #may be extended by subclasses
        return self.state.get_work()    

    def run_internal(self):
        work = None
        while True:  #loop until state.work_queue is empty and signal_stop is called
            try:
                work = self.myparent.get_work()
            except queue.Empty:
                if self.is_stop():
                    print(self.name + " is stopping")
                    break
                time.sleep(1)
                continue
            self.current_work = work
            error_msg = None
            try:
                self.loop_work(work)
            except Exception as e:
                error_msg = "Exception in " + self.name + " {}".format(e)
            except:
                error_msg = "General error in " + self.name
                
            if error_msg is None:
                self.state.work_done(self.current_work)
            else:
                self.state.work_failed(self.current_work)
                print(error_msg)
                self.log(error_msg)
            self.current_work = None
        self.state.thread_done(self)

    #static
    def run_static(this):
        this.run_internal()

    def start(self):
        self.t = Thread(target=self.run_internal, args=() )
        self.t.start()
       
        