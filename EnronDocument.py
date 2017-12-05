# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 19:45:58 2016

@author: neerbek
"""

import os
import io
import nltk  # type: ignore


class EnronDocument:
    def __init__(self):
        self.emailid = None
        self.docs = {}

    # static
    def get_parent_id(fileid):
        index = fileid.find(".", -4)
        if (index == -1):
            return fileid
        return fileid[:index]

    def get_document_count(db):
        count = 0
        for d in db.values():
            count += len(d.docs)
        return count

    def add_doc(self, emailid, docid, docpath):
        if self.emailid != None and emailid != self.emailid:
            raise Exception("Emailid does not match! {} != {}".format(
                emailid, self.emailid))
        self.emailid = emailid  # overwritten everytime - a bit ugly
        self.docs[docid] = docpath

    def add_doc_from_path(self, docpath):
        d, fname = os.path.spilt(docpath)
        fileid = fname[:fname.index(".txt")]
        emailid = EnronDocument.EnronDocument.get_parent_id(fileid)
        self.add_doc(emailid, fileid, docpath)


class EnronText:
    def __init__(self, enron_label, filepath):
        self.enron_label = enron_label
        self.filepath = filepath
        self.text = None

    def load_text(self):
        self.text = ""
        with io.open(self.filepath, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                self.text += line + "\n"

    def get_sentences(self):
        if self.text is None:
            raise Exception("EnronText, file has not been read")
        return nltk.tokenize.sent_tokenize(self.text)


class EnronLabel:
    def __init__(self, problem=None, fileid=None, strata=None, relevance=None):
        self.set_label(problem, fileid, strata, relevance)

    def set_label(self, problem, fileid, strata, relevance):
        self.problem = problem
        self.fileid = fileid
        self.strata = strata
        self.relevance = relevance


# Topics reworded a bit
# 201. All documents which relate to the Company’s engagement in structured commodity transactions known as "prepay transactions."
# 202. All documents which relate to the Company’s engagement in transactions that the Company characterized as compliant with FAS 140 (or its predecessor FAS 125).
# 203. All documents which relate to whether the Company had met, or could, would, or might meet its financial forecasts, models, projections, or plans at any time after January 1, 1999.
# 204. All documents which relate to any intentions, plans, efforts, or activities involving the alteration, destruction, retention, lack of retention, deletion, or shredding of documents or other evidence, whether in hard - copy or electronic form.
# 205. All documents which relate to energy schedules and bids, including but not limited to, estimates, forecasts, descriptions, characterizations, analyses, evaluations, projections, plans, and reports on the volume(s) or geographic location(s) of energy loads.
# 206. All documents which relate to any discussion(s), communication(s), or contact(s) with financial analyst(s), or with the firm(s) that employ them, regarding (i) the Company’s financial condition, (ii) analysts’ coverage of the Company and/or its financial condition, (iii) analysts’ rating of the Company’s stock, or (iv) the impact of an analyst’s coverage of the Company on the business relationship between the Company and the firm that employs the analyst.
# 207. All documents which relate to fantasy football, gambling on football, and related activities, including but not limited to, football teams, football players, football games, football statistics, and football performance
# The  8th  topic  (number  200)  was  a  new  one regarding real estate, with no background complaint, though it had 7 sentences of guidelines on what was responsive  or  not.

class EnronLabels:
    def __init__(self):
        # self.problems = {"200":[], "201":[], "202":[], "203":[], "204":[], "205":[], "206":[], "207":[] }
        self.problems = {}

    def add_label(self, problem, fileid, strata, relevance):
        if problem not in self.problems:
            self.problems[problem] = []
        l = self.problems[problem]
        label = EnronLabel(problem, fileid, strata, relevance)
        l.append(label)

    def get_labels(self, problem):
        p_list = self.problems[problem]
        pos = []
        neg = []
        not_rated = []
        for p in p_list:
            if p.relevance == 1:
                pos.append(p)
            elif p.relevance == 0:
                neg.append(p)
            else:
                not_rated.append(p)
        return pos, neg, not_rated
