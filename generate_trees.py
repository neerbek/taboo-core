#  -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:54:56 2017

@author: neerbek
"""

import os
import sys

from numpy.random import RandomState

import server_enron_helper
import rnn_enron
import EnronDocument
import server_rnn_helper
import similarity.load_trees as load_trees
import ai_util

startindex = 0
endindex = 5
outputfile = None
trec_path = "/home/neerbek/jan/AIProjectsData/TREC2010"
# inputfile = "/home/neerbek/jan/AIProjectsData/TREC2010/data/corpora/trec/legal10/emails/edrm-enron-v2_kaminski-v_xml_1of2.zip/text_000/3.317398.KFV32XJXBEP5TEF42WHM3J1JO0BHARTVB.1.txt"
inputfile = None
problem_label = "201"
sentence_length = 300  # 150 is 10x faster(!)
totaltimer = ai_util.Timer("Total time: ")
treetimer = ai_util.Timer("Generating trees time: ")
glove_path = "../code/glove"
totaltimer.begin()


def syntax():
    print("""syntax: generate_trees.py [-startindex <index>]
[-endindex <index>][-outputfile <file>][-help][-sentence_length <length>][-problem_label <label>]
    [-inputfile <file>][-glove_path <path>][-trec_path <path>[-h | --help | -?]
""")
    sys.exit()


argn = len(sys.argv)
i = 1
if argn == 1:
    syntax()

while i < argn:
    setting = sys.argv[i]
    arg = None
    if i < argn - 1:
        arg = sys.argv[i + 1]

    next_i = i + 2
    if setting == '-startindex':
        startindex = int(arg)
    elif setting == '-endindex':
        endindex = int(arg)
    elif setting == '-outputfile':
        outputfile = arg
    elif setting == '-problem_label':
        problem_label = arg
    elif setting == '-inputfile':
        inputfile = arg
    elif setting == '-sentence_length':
        sentence_length = int(arg)
    elif setting == '-glove_path':
        glove_path = arg
    elif setting == '-trec_path':
        trec_path = arg
    else:
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        else:
            syntax()
            raise Exception("unknown option: " + setting)
        next_i = i + 1
    i = next_i

labelfile = os.path.join(trec_path, "data/corpora/trec/legal10-results/labels/qrels.t10legallearn")
document_root = os.path.join(trec_path, "data/corpora/trec/legal10/emails")

# we need to load glove vectors, but we don't need them..."
nx = 50
rng = RandomState(1234)
LT = rnn_enron.get_word_embeddings(
    os.path.join(glove_path, "glove.6B.{}d.txt".format(nx)), rng, 10000)

rnn_enron.MAX_SENTENCE_LENGTH = sentence_length
# rnn_enron.DEBUG_PRINT = True

enronTexts = None
if inputfile == None:  # load all
    allEnronTexts = server_enron_helper.load_labeled_documents(
        labelfile, document_root, problem_label)
    enronTexts = allEnronTexts[startindex:endindex]
else:  # load only one (for debugging)
    enronLabel = EnronDocument.EnronLabel(
        problem=problem_label, fileid=inputfile, strata=1000, relevance="0")
    enronText = EnronDocument.EnronText(enronLabel, inputfile)
    enronTexts = [enronText]
    enronTexts = server_enron_helper.load_text(enronTexts)
    #    sentences = enronTexts[0].get_sentences()
    #    i=2
    #    sentences[i] = sentences[i].replace('#', '')
    #    sentences[i] = sentences[i].replace('\x0b', '')
    #    sentences[i] = re.sub(r'[\x0e-\x1f]', r'', sentences[i]) #remove non-ascii
    #    sys.exit()
    print("sentences:")
    for i, sentence in enumerate(enronTexts[0].get_sentences()):
        print(i, sentence)

parserStatistics = rnn_enron.ParserStatistics()
treetimer.begin()
trees = server_rnn_helper.get_trees(enronTexts, LT, parserStatistics)
treetimer.end()

if outputfile == None:
    outputfile = "trees_{}_{}_{}_{}.txt".format(problem_label, sentence_length, startindex, endindex)
    print("saving to file " + outputfile)
load_trees.put_trees(outputfile, trees)  # save trees

totaltimer.end()
totaltimer.report()
treetimer.report()
print(
    "Trees has been loaded. We had {} docs. This generated {} sentences. We split long sentences {} times. Total number of splits was {}".
    format(
        len(enronTexts), parserStatistics.sentences,
        parserStatistics.sentencesToBeSplit, parserStatistics.splits))
print(
    "We received {} empty splitted trees and {} empty sentence trees. Number of trees ignored because the parsing changed the text was {}".
    format(parserStatistics.emptySubTrees, parserStatistics.emptySentenceTrees,
           parserStatistics.failedSameSentences))

print("***trees loaded")
