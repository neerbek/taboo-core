# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:19:30 2017

@author: neerbek
"""

import os
import io

from numpy.random import RandomState
import time
import re
from collections import defaultdict

import server_rnn
import inference_enron
import EnronDocument

import sys
sys.setrecursionlimit(10000)


class ServerState:
    def __init__(self, max_embedding_count=-1, glove_path="../code/glove/"):
        self.enronLabels = None
        self.rnn = None
        self.rng = RandomState(1234)
        self.server_rnn_state = None
        self.server_rnn_state = server_rnn.State(
            max_embedding_count=max_embedding_count, glove_path=glove_path)

    def load_model(self):
        self.rnn = server_rnn.RNNWrapper()
        # server_rnn.nn_model.load_v1(self.rnn.rnn, 'model_rootacc0.7469.v1.save')
        # self.rnn.save('model_rootacc0.7469.v2.save')
        self.rnn.load('models/model_rootacc0.7469.v2.save')

    def initialize(self, max_count=-1):
        if self.rnn == None:
            trainer = server_rnn.Trainer()
            self.server_rnn_state.load_trees(trainer, max_count)
            self.rnn = server_rnn.RNNWrapper()


class KeywordState:
    def __init__(self):
        self.word_counts = None
        self.yes_weights = None
        self.no_weights = None
        self.indicators = None

    def load_model(self, trees, cut_off):
        self.word_counts = inference_enron.get_word_counts(trees)
        (yes_weights,
         no_weights) = inference_enron.get_weights(self.word_counts)
        self.yes_weights = yes_weights
        self.no_weights = no_weights
        self.indicators = inference_enron.get_indicators(cut_off, yes_weights)

    def initialize(self, rnn_server_state, cut_off=0.7):
        if self.word_counts == None:
            self.load_model(rnn_server_state.train_trees, cut_off)


class Prediction:
    def __init__(self, beginIndex, endIndex, prediction, text=""):
        self.beginIndex = beginIndex
        self.endIndex = endIndex
        self.prediction = int(prediction)
        self.text = text


class Predictions:
    def __init__(self, predictions):
        self.predictions = predictions


def index_sentences_to_predictions(indexed_sentences):
    res = []
    for i in range(len(indexed_sentences)):
        s = indexed_sentences[i]
        res.append(Prediction(s.beginIndex, s.endIndex, s.pred, s.sentence))
    # print(res)
    return Predictions(res)


# labelfile="/home/neerbek/Dropbox/DLP/trec/legal10-results/labels/qrels.t10legallearn"
def load_labels(labelfile):
    db = EnronDocument.EnronLabels()
    count = 0
    with io.open(labelfile, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            colonindex = line.index(":")
            problem = line[:colonindex]
            spaceindex1 = line.index(" ")
            fileid = line[colonindex + 1:spaceindex1]
            spaceindex2 = line.index(" ", spaceindex1 + 1)
            strata = line[spaceindex1 + 1:spaceindex2]
            strata = int(strata)
            relevanceS = line[spaceindex2 + 1:]
            relevance = int(relevanceS)
            db.add_label(problem, fileid, strata, relevance)
            count += 1
            if count % 100000 == 0:
                print("added {} labels".format(count), relevanceS)
                # file has 5M labels (8 questions x 0.65M labels)
    print("Done. added {} labels".format(count))
    return db


def load_doclist(dirname):  # TODO: does not work for relative paths...
    emaildir = dirname
    enronDocuments = defaultdict(
        lambda: EnronDocument.EnronDocument())  # url to EnronDocument
    for dirName, subdirList, fileList in os.walk(emaildir):
        d = os.path.join(emaildir, dirName)
        count = 0
        for i, fname in enumerate(fileList):
            count += 1
            fileid = fname[:fname.index(".txt")]
            if fileid in enronDocuments and fileid in enronDocuments[fileid].docs:
                raise Exception("Fileid found twice! {} and {}".format(
                    enronDocuments[fileid].emailid, fname))
            emailfileid = EnronDocument.EnronDocument.get_parent_id(fileid)
            enronDocuments[emailfileid].add_doc(
                emailid=emailfileid,
                docid=fileid,
                docpath=os.path.join(d, fname))
        print_status = True
        if count == 0 and dirName[-4] == '.':  # top directories end with ".zip" or ".pst" and are expected to be empty
            print_status = False
        if print_status:
            print('Found directory: ' + dirName)
            print('   In directory: #files={}'.format(count))
    return enronDocuments


def get_enron_documents(documents, label_list, enronDocuments):
    for enronLabel in label_list:
        emailid = EnronDocument.EnronDocument.get_parent_id(enronLabel.fileid)
        filepath = enronDocuments[emailid].docs[enronLabel.fileid]
        documents.append(EnronDocument.EnronText(enronLabel, filepath))


def get_indexes(text, char):
    indexes = [i for i, c in enumerate(text) if c == char]
    return indexes


def keep_doc2(i, d):
    # keep documents that only contain header as subject or person might be tricker for labeling
    if len(d.text) < 10:
        return False
    if len(get_indexes(d.text, "\n")) <= 3 and (
            d.text.startswith("URL") or d.text.startswith("Attachment") or
            d.text.startswith("[InternetShort")):
        return False
    if len(d.text) > 500:
        count_tabs = d.text.count('\t')
        if count_tabs > 10 and count_tabs > d.text.count('\n'):
            return False  # more tabs than newlines, this is probabably a spreadsheet
    return True


def replace(text, key, value):
    text = text.replace(" " + key + " ", " " + value + " ")
    text = text.replace("\n" + key + " ", "\n" + value + " ")
    text = text.replace(" " + key + "\n", " " + value + "\n")
    return text


def clean_text(text, look_for_header=None):
    # don't include footer
    pattern = "***********"
    len_pattern = len(pattern)
    index = text.find(pattern)
    if index != -1:
        index2 = text.find(pattern, index + len_pattern)
        if index2 != -1:
            if text[index + len_pattern + 1:index2].startswith(
                    "EDRM Enron Email Data Set has been produced in EML, PST and NSF format by ZL Technologies"
            ) and index2 - index >= 334 and index2 - index < 350:
                text = text[:index] + text[index2 + len_pattern:]
    # remove email headers, but include subject
    # also remove lines with only email addresses
    s = text.split("\n")
    s2 = []
    is_header = look_for_header
    header_label = ""
    for l in s:
        if len(l) == 0:
            is_header = False
            continue
        l = l.replace("[IMAGE]", "")
        l = l.strip()
        if is_header is None:  # don't know if this is email with headers, we check first line
            index = l.find(" ")
            if index > 0 and l[index - 1] == ':':
                is_header = True
            else:
                is_header = False
        include = True
        if is_header:
            index = l.find(" ")
            if index > 0 and l[index - 1] == ':':
                header_label = l[:index - 1]
                l = l[index:]
                l = l.strip()
            if header_label == "Subject" or header_label == "subject":
                include = True
            else:
                include = False
        elif len(l) < 40:
            if l.startswith("<") and (l.find("@ENRON.com") != -1 or
                                      l.find("@cwt.com") != -1 or
                                      l.find("@enron.com>") != -1):
                include = False
            elif l.startswith("\"") and l.endswith("(E-mail)\""):
                include = False
            elif l.startswith("\"") and l.endswith("\""):
                l = l[1:-1]
            elif l == "_______________________":
                include = False
        elif l == "==============================================================================":
            include = False
        elif l == "**********************************************************************":
            include = False
        elif l.startswith("Attachment: "):
            include = False
        if include:
            s2.append(l)
    text = "\n".join(s2)
    # nltk has problems with appreviations with dot, so we remove
    text = replace(text, "Mr.", "Mr")
    text = replace(text, "Mrs.", "Mrs")
    text = replace(text, "Ms.", "Ms")
    text = replace(text, "i.e.", "ie")
    text = replace(text, "Inc.", "Inc")
    text = replace(text, "U.S.", "US")
    text = replace(text, "P.O.", "PO")
    text = replace(text, "No.", "No")
    text = replace(text, "A.", "A:")
    text = replace(text, "B.", "B:")
    # nltk has problems with '
    text = text.replace("'", "")
    text = text.replace("\"\"", "\"")
    text = text.replace("\"", "")

    # add punctuation in emails where \n is used as punctuation. E.g.
    # if we have newline and last char was not punctuation and
    # next char isuppper -> then add '.'
    index = 1
    index = text.find("\n", index)
    length = len(text)
    while index != -1 and index <= length - 2:
        c = text[index - 1]
        if c != '.' and c != ',' and c != ';' and c != ':' and c != '\n':
            c = text[index + 1]
            if c.isupper():
                text = text[:index] + ".\n" + text[index + 1:]
                length += 1
                index += 1
        index = text.find("\n", index + 1)
    return text


def load_text(enronTexts):
    start = time.time()
    for i, d in enumerate(enronTexts):
        d.load_text()
        d.text = re.sub(r'[^\x09-\x7f]', r'', d.text)  # remove non-ascii
        d.text = d.text.replace('\x0b', '')            # remove non-ascii
        d.text = re.sub(r'[\x0e-\x1f]', r'', d.text)   # remove non-ascii
        # keeping \x0a (newline), \x0c (form-feed), \x0d (carriage return)
        # next ascii after \x1f is \x20 (space)
        d.text = d.text.replace('#', '')

        if i % 200 == 0:
            print(i, "time elapsed is: {}".format(time.time() - start))
            # load_time = 0
            # split_time=0
    doc2 = [d for i, d in enumerate(enronTexts) if keep_doc2(i, d)]
    for d in doc2:
        d.text = clean_text(d.text)
    print("EnronTexts read. time elapsed is:", time.time() - start)
    return doc2


def remove_unfound_files(doclist, l):
    # removing unknown emails (because we are working on a debug subset)
    new_label_list = list()
    for d in l:
        emailid = EnronDocument.EnronDocument.get_parent_id(d.fileid)
        if emailid in doclist and d.fileid in doclist[emailid].docs:
            new_label_list.append(d)
    if len(l) != len(new_label_list):
        print("removed {} unfound documents!".format(
            len(l) - len(new_label_list)))
    else:
        print("All labeled files found!")
    return new_label_list


def load_labeled_documents(labelfile, document_root, problem_label):
    # labels = load_labels("/home/neerbek/Dropbox/DLP/trec/legal10-results/labels/qrels.t10legallearn")
    # labels = load_labels("enron_labels.txt")
    print('loading labels')
    labels = load_labels(labelfile)

    print('getting labeled docs')
    pos, neg, not_rated = labels.get_labels(problem_label)
    enronDocumentDict = load_doclist(document_root)

    # removing unknown emails (because we are working on a debug subset)
    pos = remove_unfound_files(enronDocumentDict, pos)
    neg = remove_unfound_files(enronDocumentDict, neg)
    enronTexts = []
    get_enron_documents(enronTexts, pos, enronDocumentDict)
    get_enron_documents(enronTexts, neg, enronDocumentDict)
    enronTexts2 = load_text(enronTexts)
    if len(enronTexts) != len(enronTexts2):
        print(
            "Removed {} documents from original list of size {}. Removed documents were very short".
            format(len(enronTexts) - len(enronTexts2), len(enronTexts)))

    for i in range(len(enronTexts2)):
        d = enronTexts2[i]
        if d.enron_label.relevance == 0:
            d.enron_label.relevance = "0"
        elif d.enron_label.relevance == 1:
            d.enron_label.relevance = "4"
        else:
            raise Exception(
                "unknown label encountered {}".format(d.enron_label.relevance))

    rand = RandomState(374637)
    rand.shuffle(enronTexts2)
    return enronTexts2
