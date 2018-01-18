# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:46:14 2017

@author: neerbek
"""

from stat_parser.learn import build_model  # type: ignore
from stat_parser.tokenizer import PennTreebankTokenizer  # type: ignore
# from stat_parser.word_classes import is_cap_word
from stat_parser.parser import CKY  # type: ignore
from nltk import Tree  # type: ignore
import similarity.load_trees as load_trees  

# pcfg + treebanks http://courses.cs.washington.edu/courses/cse517/13wi/slides/cse517wi13-Parsing.pdf


class StatisticTextParser:
    def __init__(self):
        self.pcfg = build_model()
        self.tokenizer = PennTreebankTokenizer()

    def nltk_tree(self, t):
        return Tree(t[0], [c if isinstance(c, str) else self.nltk_tree(c) for c in t[1:]])

    # sentence=l2
    # sentence=sent
    def norm_parse(self, sentence, timers):
        words = self.tokenizer.tokenize(sentence)
        # words
        # if is_cap_word(words[0]):
        #    words[0] = words[0].lower()

        norm_words = []
        for word in words:
            if isinstance(word, tuple):
                # This is already a word normalized to the Treebank conventions
                raise Exception("Not supported")
                # norm_words.append(word)
            else:
                # rare words normalization
                norm_words.append((self.pcfg.norm_word(word), word))
        timers.ckyTimer.begin()
        cky = CKY(self.pcfg, norm_words)
        timers.ckyTimer.end()
        return cky

    # sent = " 9."
    # sent = l2
    # self = parser
    def get_tree(self, sent, timers):
        "Returns a similarity.load_trees.Tree for the sentence sent. We use the prob. parser model from stat_parser"
        timers.normTimer.begin()
        tree = self.norm_parse(sent, timers)
        timers.normTimer.end()
        # print(tree)
        if tree is None:
            return None
        timers.nltkTimer.begin()
        ntree = self.nltk_tree(tree)
        timers.nltkTimer.end()
        line = " " + ntree.__str__()  # serialize ntree and add ' ' at start
        line = " " + ntree._pformat_flat(nodesep='', parens='()', quotes=False)
#        line = line.replace("\n","")     #remove newlines
#        line2 = line.replace("  "," ")
#        while (line2!=line):             #remove extra spaces
#            line = line2
#            line2 = line.replace("  "," ")
        if not line.startswith(" ("):
            raise Exception("train parser" +
                            " line does not start with \" (\"")
        t = load_trees.Node(None)
        # print("Line is " + line)
        i = load_trees.parse_line(line, 2, t)
        if i < len(line) - 1:  # Not all of line parsed
            raise Exception(
                "train parser" + " parsing line failed. There was more than one tree in the line. {}".format(i))
        l2 = load_trees.output(t)
        if l2 != line:  # Lines differ
            raise Exception(
                "train parser" + " marshall and unmarshalling differs" + "\n" + line + "\n" + l2)
        if not t.is_binary():
            raise Exception("train parser" + " tree is not binary")
        if not t.has_only_words_at_leafs():
            raise Exception("train parser" +
                            " tree is not properly normalized")
        return t

# p = StatisticTextParser()
# t = p.get_tree("Jan went to the lake. He looked at all the funny fish")
