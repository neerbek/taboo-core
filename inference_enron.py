# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:27:38 2016

@author: neerbek
"""

import io
from collections import defaultdict

import similarity.load_trees as load_trees
from server_rnn_helper import IndexSentence


def normalize_word(w):
    w = w.lower()
    #     if w=="as":
    #         return w
    #     if w.endswith("ies"):
    #         w = w[:-3] + "y"
    #     elif w.endswith("s"):
    #         w = w[:-1]
    #     elif w.endswith("ed"):
    #         w = w[:-2]
    #     #    elif w.endswith("paid"):
    #     #        w = w[:-2] + "y"
    return w


def tree_contains_word(node, w):
    """ outputs true if w is in tree (normalized)"""
    if (node == None):
        return False
    if (node.word != None) and normalize_word(node.word) == w:
        return True
    return tree_contains_word(node.left, w) or tree_contains_word(
        node.right, w)


def tree_contains_words(node, s):
    """ outputs true if any of s is in tree (normalized)"""
    if (node == None):
        return False
    if (node.word != None) and normalize_word(node.word) in s:
        return True
    return tree_contains_words(node.left, s) or tree_contains_words(
        node.right, s)


def output_word_counts(node, word_counts):
    """ outputs the words in the node and it's children"""
    if (node == None):
        return
    output_word_counts(node.left, word_counts)
    if (node.word != None):
        w = normalize_word(node.word)
        word_counts.add(w)
    output_word_counts(node.right, word_counts)


# def split_trees(trees):
#     y = []
#     n = []
#     for t in trees:
#         if (t.syntax=="0"):
#             n.append(t)
#         else:
#             y.append(t)
#     return (y,n)
#
# (yes_trees, no_trees) = split_trees(trees)
# len(yes_trees), len(no_trees) #Out[54]: (2985, 6015)


def get_word_counts(trees):
    word_counts = {}
    for t in trees:
        tmp = set()
        output_word_counts(t, tmp)
        for w in tmp:
            if not (w in word_counts):
                word_counts[w] = {}
                word_counts[w]["0"] = 0
                word_counts[w]["4"] = 0
            if t.syntax == "1":
                word_counts[w]["4"] += 1
            else:
                word_counts[w][t.syntax] += 1
    return word_counts


def get_weights(word_counts, supportCutoff=9):
    yes_weights = defaultdict(list)
    no_weights = defaultdict(list)
    for w in word_counts.keys():
        yc = word_counts[w]["4"]
        nc = word_counts[w]["0"]
        if (yc + nc) < supportCutoff:
            continue
        if (yc > nc):
            yes_weights[yc / (yc + nc)].append(w)
        else:
            no_weights[nc / (yc + nc)].append(w)
    return (yes_weights, no_weights)


def get_indicators(confidence, weights):
    """Takes a confidence number and a map from weights (confidences) to words with that weight. Returns a set of all words which have a confidence higher that <confidence>"""
    sorted_weights = weights.keys()
    sorted_weights = sorted(sorted_weights, reverse=True)
    indicators = set()
    for w in sorted_weights:
        if w > confidence:
            for word in weights[w]:
                indicators.add(word)
    return indicators


def get_accuracy(ttrees, indicators, logInstanceDetails=False):
    acc = 0
    count = 0
    # = ttrees[0]
    # indicators
    for t in ttrees:
        is_sensitive = tree_contains_words(t, indicators)
        count += 1
        tree_is_sensitive = (t.syntax == "4" or t.syntax == "1")
        if tree_is_sensitive == is_sensitive:  # e.g. is label equal to prediction
            acc += 1
        if logInstanceDetails:
            node_count = load_trees.count_nodes(t)
            text = load_trees.output_sentence(t)
            tree_str = load_trees.output(t)
            print(count, tree_is_sensitive, (tree_is_sensitive == is_sensitive), "N/A", node_count, text, tree_str)

    return acc / count


def get_confusion_numbers(ttrees, indicators):
    tp = 0  # true == sensitive
    fp = 0
    tn = 0
    fn = 0
    count = 0
    for t in ttrees:
        is_sensitive = tree_contains_words(t, indicators)
        count += 1
        if (t.syntax == "4" or t.syntax == "1"):
            if is_sensitive:
                tp += 1
            else:
                fn += 1
        else:
            if is_sensitive:
                fp += 1
            else:
                tn += 1
    return (tp, fp, tn, fn)


def get_predictions(text, indicators):
    e = text.split()
    index = 0
    sent = IndexSentence(0, "")
    sent.pred = False
    cur = text
    res = []
    for w in e:
        is_sensitive = (w in indicators)
        if is_sensitive != sent.pred:
            sent.sentence = text[sent.beginIndex:index]
            sent.endIndex = index
            res.append(sent)
            sent = IndexSentence(index, w)
            sent.pred = is_sensitive
        i = cur.find(w) + len(w)
        cur = cur[i:]
        index += i
    sent.sentence = text[sent.beginIndex:index]
    sent.endIndex = index
    res.append(sent)
    return res


#  conf | acc
#   0.6 | 0.7119
#   0.7 | 0.7441
#   0.8 | 0.7378
#  0.65 | 0.7406
#  0.75 | 0.7434
#  0.725 | 0.7420
#  0.675 | 0.7476
#  0.6875 | 0.7461
#  0.676 | 0.7476
#  0.674 | 0.7476
def save_report(output, filename="report_inference.txt"):
    with io.open(filename, 'w', encoding='utf8') as f:
        for t in output:
            f.write(t + "\n")


def list_best_non_sensitive_keywords(no_weights, word_counts):
    k = no_weights.keys()
    k = sorted(k, reverse=True)
    k = sorted(k)
    print("Best no'es")
    for i in range(10):
        for w in no_weights[k[i]]:
            print(i, k[i], w, word_counts[w]["0"])


if __name__ == "__main__":
    trees = load_trees.get_trees("trees/train.txt")
    word_counts = get_word_counts(trees)
    print(len(word_counts.keys()))  # 9260 (normalized) 10463 (lowercase), 13083
    (yes_weights, no_weights) = get_weights(word_counts)
    k = yes_weights.keys()
    k = sorted(k, reverse=True)
    len(yes_weights), len(no_weights)
    print("Best yes'es")
    for i in range(10):
        for w in yes_weights[k[i]]:
            print(i, k[i], w, word_counts[w]["4"])

    indicators = get_indicators(0.7, yes_weights)
    len(indicators)
    "Transaction" in indicators
    s = " As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact ."
    s = "Attached are the credit worksheets for Royal Bank of Canada and Bow River Trust As we discussed we have proposed letters of credit for the approved form of collateral pending further discussion with Treasury regarding funding impact We may have to move to cash margining if necessary"
    s = " As used above , payments scheduled to be made on any date or during any period shall be determined as if all payments are to be made through and including the Scheduled Termination Date without regard to whether -LRB- a -RRB- the Windup Date occurs prior thereto and -LRB- b -RRB- the Early Termination Date has occurred ."
    e = s.split(" ")
    total_weight = 0
    count = 0
    for w in e:
        w = normalize_word(w)
        if w == "":
            continue
        weight = word_counts[w]["4"] / (
            word_counts[w]["0"] + word_counts[w]["4"])
        total_weight += weight
        print(w, weight, w, word_counts[w]["4"])
        count += 1

    print("average: ", total_weight / count)

    ttrees = load_trees.get_trees("trees/test.txt")
    indicators = get_indicators(0.675, yes_weights)
    print("accuracy: ", get_accuracy(ttrees, indicators))
