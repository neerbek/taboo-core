"""
CKY algorithm from the "Natural Language Processing" course by Michael Collins
https://class.coursera.org/nlangp-001/class
"""
from collections import defaultdict
from pprint import pprint
from typing import Dict, Tuple, Any

MAX_TIME_FORWARD = 120  # wait max 2 minutes on forward pass
try:
    from nltk import Tree

    def nltk_tree(t):
        return Tree(t[0], [c if isinstance(c, str) else nltk_tree(c) for c in t[1:]])

    nltk_is_available = True

except ImportError:
    nltk_is_available = False

from stat_parser.learn import build_model
from stat_parser.tokenizer import PennTreebankTokenizer
from stat_parser.treebanks.normalize import un_chomsky_normal_form
from stat_parser.word_classes import is_cap_word

from ai_util import Timer, TimerList


def argmax(lst):
    return max(lst) if lst else (0.0, None)


def backtrace(back, bp, timerList):
    timerList.report(min_seconds=5, update_timers=True)
    # Extract the tree from the backpointers
    if not back:
        return None
    if len(back) == 6:
        (X, Y, Z, i, s, j) = back
        return [X, backtrace(bp[i, s, Y], bp, timerList),
                backtrace(bp[s + 1, j, Z], bp, timerList)]
    else:
        (X, Y, i, i) = back
        return [X, Y]


def CKY(pcfg, norm_words):
    timerList = TimerList()
    timerList.totalTimer = Timer("total CKY time")
    timerList.addTimer(timerList.totalTimer)
    timerList.forwardTimer = Timer("forward pass")
    timerList.addTimer(timerList.forwardTimer)
    timerList.backwardTimer = Timer("backward pass")
    timerList.addTimer(timerList.backwardTimer)
    timerList.totalTimer.begin()
    x, n = [("", "")] + norm_words, len(norm_words)

    # Charts
    pi: Dict[Tuple[int, int, str], float]; pi = defaultdict(float)
    bp: Dict[Tuple[int, int, str], Tuple[Any, ...]]; bp = defaultdict(tuple)
    for i in range(1, n + 1):
        for X in pcfg.N:
            norm, word = x[i]
            if (X, norm) in pcfg.q1:
                # score of rule X which yields word norm==word
                pi[i, i, X] = pcfg.q1[X, norm]
                bp[i, i, X] = (X, word, i, i)  # rule X in back-parse

    # Dynamic program
    for l in range(1, n):
        for i in range(1, n - l + 1):
            timerList.forwardTimer.begin()
            j = i + l
            for X in pcfg.N:  # we are looking for next node X in tree. Node corresponds to rule X -> Y Z
                # Note that we only check rules that exist in training
                # and have non-zero probability
                score, back = argmax([(
                    pcfg.q2[X, Y, Z] * pi[i, s, Y] * pi[s + 1, j, Z],
                    (X, Y, Z, i, s, j)  # this becomes back
                ) for s in range(i, j)
                    for Y, Z in pcfg.binary_rules[X]  # look for rules X->Y Z
                    if pi[i, s, Y] > 0.0
                    if pi[s + 1, j, Z] > 0.0
                ])

                if score > 0.0:
                    bp[i, j, X], pi[i, j, X] = back, score  #
            timerList.forwardTimer.end()
            if timerList.report(min_seconds=5, update_timers=True):
                if timerList.forwardTimer.elapsed > MAX_TIME_FORWARD:
                    print("Forward pass is taking too long. Aborting")
                    return None
    _, top = max([(pi[1, n, X], bp[1, n, X]) for X in pcfg.N])
    timerList.backwardTimer.begin()
    return backtrace(top, bp, timerList)
# len(parser.pcfg.N)
# parser.pcfg.N


class Parser:
    def __init__(self, pcfg=None):
        if pcfg is None:
            pcfg = build_model()

        self.pcfg = pcfg
        self.tokenizer = PennTreebankTokenizer()

        if nltk_is_available:
            self.parse = self.nltk_parse
        else:
            self.parse = self.raw_parse

    def is_nltk_available(self):
        return nltk_is_available

    def norm_parse(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        if is_cap_word(words[0]):
            words[0] = words[0].lower()

        norm_words = []
        for word in words:
            if isinstance(word, tuple):
                # This is already a word normalized to the Treebank conventions
                norm_words.append(word)
            else:
                # rare words normalization
                norm_words.append((self.pcfg.norm_word(word), word))
        return CKY(self.pcfg, norm_words)

    def raw_parse(self, sentence):
        tree = self.norm_parse(sentence)
        un_chomsky_normal_form(tree)
        return tree

    def nltk_parse(self, sentence):
        return nltk_tree(self.raw_parse(sentence))


def display_tree(tree):
    if nltk_is_available:
        tree.draw()
    else:
        pprint(tree)
