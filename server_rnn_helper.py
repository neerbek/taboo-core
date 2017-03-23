# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 07:32:29 2017

@author: neerbek
"""
import nltk 

import rnn_enron
from StatisticTextParser import StatisticTextParser
from ai_util import TrainTimer


class IndexSentence:
    def __init__(self, index, sentence):
        self.beginIndex = index
        self.endIndex = index + len(sentence)
        self.sentence = sentence
        self.pred = None
        
def get_first_word(s):
    index = s.find(" ")
    if index==-1:
        return s
    return s[:index]
    
def get_indexed_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    res = []
    index = 0
    for s in sentences:
        if len(s)==0:
            continue
        w = get_first_word(s)
        i = text.find(w, index)
        if i==-1:
            raise Exception("word not found in sentence: " + w)
        res.append(IndexSentence(i,s))
        index = i + len(s)
    return res
    
def get_nltk_trees(doc_number, indexed_sentences, parserStatistics):    
    parser = StatisticTextParser()
    timers = rnn_enron.Timers()
    prev_sent_number = -2
    for sent_number, s in enumerate(indexed_sentences):
        timers.totalTimer.begin()
        tree = rnn_enron.get_nltk_parsed_tree_from_sentence(s.sentence, parser, timers, parserStatistics)
        s.tree = tree
        timers.totalTimer.end()
        if rnn_enron.DEBUG_PRINT:
            if timers.report(min_seconds=5):
                print("Doc: {}, Completed sentence {} (of {})".format(doc_number, sent_number+1, len(indexed_sentences)))
                if sent_number==prev_sent_number+1:
                    print("Long parse of sent: " + s.sentence)
                prev_sent_number = sent_number
    if rnn_enron.DEBUG_PRINT:
        timers.report()
    
    trees = []
    for s in indexed_sentences:
        if s.tree is not None:
            trees.append(s.tree)
        else:
            print("get_nltk_trees: got empty tree")
    return trees
    
def get_trees(docs, lookupTable, parserStatistics):
    trees = []
    timer = TrainTimer("****Parsed document")
    timer.begin()
    for i, d in enumerate(docs):
        text = d.text
        label = d.enron_label.relevance
        sentences = get_indexed_sentences(text)
        ttrees = get_nltk_trees(i, sentences, parserStatistics)
        if ttrees is None:
            print("Got empty result on length {} indexed sentences".format(len(sentences)))
            continue
        for t in ttrees:
            t.replace_nodenames(label)
        rnn_enron.initializeTrees(ttrees, lookupTable)
        trees.extend(ttrees)
        if rnn_enron.DEBUG_PRINT:
            timer.end()
            timer.report(i+1)
            timer.begin()
    return trees

