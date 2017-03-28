# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:51:11 2016

@author: neerbek
"""
import os
import io
import time
# from threading import Thread
from queue import Queue
import numpy as np
from numpy.random import RandomState

import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')

import similarity.load_trees as load_trees
from server_enron_helper import load_labels, load_doclist
from server_enron_helper import load_text, get_enron_documents

import util
import EnronDocument
import ParaphraseThreadState
from ParaphraseThreadState import NO_BLOCK

enron_dir = "F:/AIProjectsData/TREC2010/data/corpora/trec"
#enron_dir = "/Users/neerbek/Dropbox/DLP/trec""legal10-results/labels/qrels.t10legallearn"

def write_batch_to_file(s, index, prefix, maximum):
    util.write_file(sentences = s, filename="{}_{}.txt".format(prefix, index), maximum=maximum, do_log=False )

def get_filenames_from_length(l, batch, prefix):
    f = []
    max_batch = l // batch
    for i in range(max_batch + 1):
        f.append("{}_{}.txt".format(prefix, i))
    return f

def get_filenames(s, batch, prefix):
    return get_filenames_from_length(len(s), batch, prefix)
    
def write_file(s, batch, prefix, maximum=-1):
    l = len(s)
    max_batch = l // batch
    for i in range(max_batch + 1):
        write_batch_to_file(s[i*batch:(i+1)*batch], i, prefix, maximum= maximum)
        if i % 500 == 0:
            print("Done {}/{} batches".format(i, max_batch))


def load_idmap_impl(f):
    idmap = {}
    i = 0
    for line in f:
        line = line.strip()
        index = line.index(",")
        idmap[line[:index]] = line[index+1:]
        idmap[line[index+1:]] = line[index+1:]
        if (i%45000==0):
            print("Read {} lines".format(i))
        i += 1
    return idmap

def load_idmap(dirname, fname = "legal10/doc-uniqdoc.csv"):
    filename = os.path.join(dirname, fname)
    idmap = {}
    time1 = time.time()
    try:
        f = io.open(filename,'r',encoding='utf8')
        idmap = load_idmap_impl(f)
        time2 = time.time()
        print("done reading file[2] {}".format(time2-time1))
    finally:
        f.close()
        time2 = time.time()
        print("done closing file {}".format(time2-time1))
    return idmap
        

#idmap = load_idmap(enron_dir)
#idmap2  = load_idmap(enron_dir, "legal10/msg-uniqmsg.csv")

#i1 = load_idmap(enron_dir)
#i2 = load_idmap(enron_dir)
#a = []
#a.append(load_idmap(enron_dir))
#len(idmap)
#len(idmap2)

def count_missing(s1, s2):
    count = 0
    for i, f in enumerate(s1.keys()):
        if f not in s2:
            if count < 100:
                print("missing " +f )
            count +=1
    if i%50000==0:
        print(i)
    print("missing {}".format(count))

#count_missing(idmap,idmap2)  #missing 7132
#count_missing(idmap2,idmap)  #missing 0

doclist = load_doclist(enron_dir)

#count=0
#for f in idmap:
#    if EnronDocument.get_parent_id(f) not in doclist:
#       count += 1 

doccount = EnronDocument.EnronDocument.get_document_count(doclist)
doccount
#count
len(doclist)
#len(idmap)

# note there are 1.2M docs in idmap but only ~0.5M in our Enron corpus. 
# The labels are only given for the 0.5M corpus



labelfile = os.path.join(enron_dir, "legal10-results/labels/qrels.t10legallearn")
labels = load_labels(labelfile)

#l = [l.relevance for l in labels.problems["201"]]
#hist = np.histogram(l)
#print(hist)

#reload(EnronDocument)

#posmap = {}
#negmap = {}
#for k in labels.problems.keys():
#    pos,neg,not_rated = labels.get_labels(k)
#    print(len(pos),len(neg), len(not_rated))
#    for p in pos:
#        posmap[k + ":" + p.fileid] = p
#    for n in neg:
#        negmap[k + ":" + n.fileid] = n
#
#for p in posmap.values():
#    emailid = EnronDocument.EnronDocument.get_parent_id(p.fileid)
#    if not (emailid in doclist):
#        print("emailid not found " + p.fileid)
#        break
#    if not (p.fileid in doclist[emailid].docs):
#        print("attachmentid not found " + p.fileid)
#        break
#
#for p in negmap.values():
#    emailid = EnronDocument.EnronDocument.get_parent_id(p.fileid)
#    if not (emailid in doclist):
#        print("emailid not found " + p.fileid)
#        break
#    if not (p.fileid in doclist[emailid].docs):
#        print("attachmentid not found " + p.fileid)
#        break
#    
#len(posmap),len(negmap)
#posmap = {}
#negmap = {}

pos,neg,not_rated = labels.get_labels("201")
print(len(pos),len(neg), len(not_rated))
labels = None   #removes 2GB+ of memory

documents = []

get_enron_documents(documents, pos, doclist)        
len(documents)
get_enron_documents(documents, neg, doclist)        
len(documents)

rand = RandomState(374637)
rand.shuffle(documents)
len(documents)

documents = documents[:630]  #25% of 2720
l = [l.enron_label.relevance for l in documents]
hist = np.histogram(l)
print(hist)
len(documents)

len(documents)
doc2 = load_text(documents)

len(doc2)

start = time.time()
split_time = 0
sentences_yes = []
sentences_no = []
for i, d in enumerate(doc2):
    ltime = time.time()
    s = d.get_sentences()   #d is EnronDocument.EnronText (I think)
    split_time+= time.time()-ltime
    w = []
    for l in s: # remove empty sentences
        l.strip()
        if len(l)!=0:
            w.append(l)
    if len(w)!=0:
        #sentences.append("FILEID276467 " + d.enron_label.fileid)
        if d.enron_label.relevance == 1:
            sentences_yes.extend(w)
        else:
            sentences_no.extend(w)
    
    if i % 100 == 0:
        print(i, "time elapsed is: {}. Splitting text={}".format(time.time() - start, split_time))
        #load_time = 0
        #split_time=0
done = time.time()
print("Number of sentences: YES={}. NO={} ".format(len(sentences_yes), len(sentences_no)))
print("time elapsed is:", done - start)


def keep(s):   #for now we keep all
    return True

print(len(sentences_yes), len(sentences_no))
sentences_yes[:] = [s.replace("\n", " ").replace("\"", "").replace(":", ";") for s in sentences_yes if keep(s)]
sentences_no[:] = [s.replace("\n", " ").replace("\"", "").replace(":", ";") for s in sentences_no if keep(s)]
print(len(sentences_yes), len(sentences_no))


write_file(sentences_yes, 20, "data/sentences_yes", maximum=-1)
write_file(sentences_no, 20, "data/sentences_no", maximum=-1)

#
# sentences written to a bunch of files
# Now use Stanford java tools to get a parse tree for each sentence
#
#util.reload(ParaphraseThreadState)
class GetTrees(ParaphraseThreadState.ThreadContainer):
    def __init__(self, tname, queue_out, tstate):
        super().__init__(tname,tstate, self)
        self.queue_out = queue_out

    def get_work(self):
        if not self.state.has_more_work():
            self.do_stop()
        return self.state.get_work()
        
    def run(self, work):
        (relevance, filename) = work
        # filename = "data/sentences_no_0.txt"
        command = ["java", "-mx3500m", "-cp",  
        "../stanford/stanford-parser-full-2015-12-09/*;../stanford/stanford-parser-full-2015-12-09/.",
        "JanBinaryTree",
           filename]
        # print(' '.join(command))
        print(self.name + " get_trees called: " + filename)
        trees = []
        t = ""
        count =0
        error_msg = None
        it = util.run_command(command)
        for i,l in enumerate(it):
            s = str(l,'iso-8859-1')
            s = s.strip()
            #print(tname + " Read: " + s)
            if s.startswith("Error:"):
                error_msg = "Error: got error from java: " + s
                break
            if s.startswith("[main]"):
                continue
            if s.startswith("done ["):
                continue
            if s.startswith("Parsing "):
                continue
            if s.startswith("Parsed "):
                continue
            if len(s)==0:
                count += 1
                if count % 3 == 0:
                    print("{} completed another tree (not adding yet ({}))".format(self.name, count))
                trees.append(t)
                t = ""
                continue
            t += " " + s
        if error_msg==None:
            print("{} completed file {}. Adding {} trees".format(self.name, filename, len(trees)))
            for t in trees:
                self.queue_out.put((relevance, t))  #may block
        else:
            count = 0
            for l in it:  #read rest of command output to make sure the command stops
                count +=1
            raise Exception(error_msg)
        print("{} done with ({},{})".format(self.name, relevance, filename))
        
class ReadTrees(ParaphraseThreadState.ThreadContainer):
    def __init__(self, tname, readerTstate, getterTstate):
        super().__init__(tname,readerTstate, self)
        self.getterTstate = getterTstate

    def get_work(self):
        if not self.state.has_more_work():
            if self.getterTstate.is_all_threads_done():
                self.do_stop()
        return self.state.get_work()
        
    def run(self, work):
        count = 0
        (relevance, line) = work
        tree = load_trees.Node()
        if not line.startswith(" ("):
            raise Exception(self.name + " line does not start with \" (\"")
        tree = load_trees.Node()
        #print("Line is " + line)
        i = load_trees.parse_line(line, 2, tree)
        if i<len(line)-1: #Not all of line parsed
            raise Exception(self.name + " parsing line failed. There was more than one tree in the line. {}".format(i))
        l2 = load_trees.output(tree)    
        if l2!=line: #Lines differ
            raise Exception(self.name + " marshall and unmarshalling differs")
        tree.replace_nodenames(relevance)
        tree.simplify()
        if not tree.is_binary():
            raise Exception(self.name + " tree is not binary")
        count +=1
        if count%10 == 0:
            print("Extracted: ", count)

len_yes = len(sentences_yes)
len_no = len(sentences_no)
print(len_yes, len_no)
#24881 112268
#Out[21]: (3967, 22806)

#copy from here
queue = Queue()

#tstate = ParaphraseThreadState.ParaphraseThreadState()
#for f in get_filenames_from_length(3967, 1000, "sentences_yes"):
#    tstate.put_work(("4", f))
#for f in get_filenames_from_length(22806, 1000, "sentences_no"):
#    tstate.put_work(("0", f))

tstate = ParaphraseThreadState.ParaphraseThreadState()
#a = get_filenames_from_length(22806, 20, "data/sentences_no")
#for f in a[0:4]:
#    tstate.put_work(("0", f))
a = get_filenames_from_length(3967, 20, "data/sentences_yes")
for f in a[0:1]:
    tstate.put_work(("4", f))
    
i=0
for i in range(1):
    tname = "New-Thread[{}]".format(i)
    t = GetTrees(tname = tname, queue_out = queue, tstate = tstate)
    #t = Thread(target=get_trees, args=(tname, queue, tstate) )
    tstate.add_thread(tname, t)


tstate.start_threads()

len(tstate.threads)



readerTstate = ParaphraseThreadState.ParaphraseThreadState()
readerTstate.work = queue

tname = "reader-thread3"
t = ReadTrees(tname, readerTstate, tstate)
readerTstate.add_thread(tname, t)
readerTstate.start_threads()


#to here

trees =  []
all_trees = []


all_trees.extend(trees)
len(all_trees)
len(trees)
trees =  []

def get_tree():
    if not readerTstate.completed_work.empty():
        l = readerTstate.completed_work.qsize()
        print("number of trees", l)
        for i in range(l):
            t = readerTstate.completed_work.get(NO_BLOCK)
            trees.append(t)
    else:
        print("no new trees")
    print("Number of trees: {}, number of errors {}, number of getter errors {}".format(len(trees), readerTstate.failed_work.qsize(),tstate.failed_work.qsize()))

get_tree()

readerTstate.completed_work.qsize()
readerTstate.has_more_work()
readerTstate.failed_work.qsize()
readerTstate.is_all_threads_done()
tstate.is_all_threads_done()
len(tstate.threads)
tstate.threads_done.qsize()
tstate.work.qsize()
tstate.completed_work.qsize()

readerTstate.threads_done.qsize()
#util.reload(similarity.load_trees)
while not tstate.work.empty():
        tstate.work.get(False)
#queue_in.qsize()


rand = RandomState(964395)
rand.shuffle(trees)  #because of threads this is not predictable

def write_data(d, filename):
    tr = []
    for t in d:
        tr.append(load_trees.output(t))
    util.write_file(tr, filename)
    
write_data(all_trees, "data/data_400_400.txt")
