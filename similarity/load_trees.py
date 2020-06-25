# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:36:32 2016

@author: neerbek
"""
import io
import ai_util

DEBUG_PRINT = False


class Warnings:
    warnRootNodeWord = 0


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.word = None
        self.syntax = None
        self.representation = None
        self.left = None
        self.right = None
        self.label = None
        self.update_count = -1

    def clone(self, node):
        # update clone_tree also
        self.parent = node.parent
        self.word = node.word
        self.syntax = node.syntax
        self.representation = node.representation
        self.left = node.left
        self.right = node.right
        self.label = node.label
        self.update_count = node.update_count

    def add_child(self):
        if (self.left == None):
            self.left = Node(self)
            return self.left
        elif self.right == None:
            self.right = Node(self)
            return self.right
        raise Exception("max two children per Node")

    def number_of_children(self):
        count = 0
        if self.left != None:
            count += 1
        if self.right != None:
            count += 1
        return count

    def is_leaf(self):
        return (self.number_of_children() == 0)

    def replace_nodenames(self, newname):
        self.syntax = newname
        if self.left != None:
            self.left.replace_nodenames(newname)
        if self.right != None:
            self.right.replace_nodenames(newname)

    def simplify(self):
        # removes nodes which only have 1 child
        if self.left != None:
            self.left.simplify()
        if self.right != None:
            self.right.simplify()

        if self.number_of_children() != 1:
            return

        if self.word != None:
            raise Exception(
                "cannot simplify tree with words in intermidiate nodes")

        if self.left != None:
            self.clone(self.left)
        elif self.right != None:
            self.clone(self.right)

    def is_binary(self):
        if self.number_of_children() == 0:
            return True
        if self.number_of_children() == 1:
            return False
        if self.number_of_children() == 2:
            return (self.left.is_binary() and self.right.is_binary())
        return False  # more than 2 children

    def has_only_words_at_leafs(self):
        if self.number_of_children() == 0:
            return (
                self.word != None
            )  # a self.word==None for leaf nodes is also a reason for reporting false
        if self.word != None:
            if self.parent != None:  # workaround for "hack" for monsanto
                return False
        return (self.left.has_only_words_at_leafs() and self.right.has_only_words_at_leafs())

    def __str__(self):
        return "Node(word: {} syntax: {} label:{} repr: {})".format(self.word, self.syntax, self.label, self.representation)


def clone_tree(node, parent=None):
    if node == None:
        return None

    n = Node()
    n.parent = parent
    n.word = node.word  # expected immutable string or int
    n.syntax = node.syntax  # expected immutable string or int
    n.representation = node.representation  # if this is an array we might need to copy this
    n.left = clone_tree(node.left, n)
    n.right = clone_tree(node.right, n)
    n.update_count = node.update_count  # expected immutable string or int
    return n


def as_array(node):
    if node == None:
        return []
    res = [node]
    res.extend(as_array(node.left))
    res.extend(as_array(node.right))
    return res


def read_rep(l, index, node):
    """ read l from index and until ] is encountered, return floats read as a rep array"""
    i = index
    rep = []
    word = ""
    while i < len(l):
        c = l[i]
        i += 1
        if (c == ' ' or c == ']'):
            if len(word) > 0:
                f = float(word)  # may throw
                rep.append(f)
                word = ""
            if c == ']':
                break
        else:
            word += c
    node.representation = rep
    return i


def output(node):
    """ outputs a string with the textual representation of the node and it's children"""
    if (node == None):
        return ""
    if node.syntax is None:
        o = " (" + "TabooNone"
        print("Warning: outputting TabooNone")
    else:
        o = " (" + node.syntax
    if (node.word != None):
        o += " " + node.word
    return o + output(node.left) + output(node.right) + ")"


def output_sentence_impl(node):
    """ outputs the words in the node and it's children"""
    if (node == None):
        return ""
    o = output_sentence(node.left)
    w = node.word
    if (w != None):
        if w == '.' or w == ',' or w == ':' or w == ';':
            o += w
        else:
            o += " " + w
    o += output_sentence(node.right)
    return o


def output_sentence(node):
    """ outputs the words in the node and it's children"""
    if (node == None):
        return ""
    o = output_sentence_impl(node)
    # o = unescape_sentence(o)
    # for the python parser it works best if we use -LRB- and -RRB- (i.e. escaped sentences)
    return o


def output_lenrep(node):
    if (node == None):
        return ""
    o = " (" + node.syntax
    o += ":{}".format(len(node.representation))
    if (node.word != None):
        o += " " + node.word
    return o + output_lenrep(node.left) + output_lenrep(node.right) + ")"


def parse_line(l, index, node):
    i = index
    # read node name ("syntaxname")
    syntaxname = ""
    while i < len(l):
        c = l[i]
        i += 1
        if (c == ' '):
            break
        syntaxname += c
    if DEBUG_PRINT and len(syntaxname) == 0:
        print("found node with no syntax name. Assuming leaf node.")
    # sep =syntaxname.find(":")
    #    if sep!=-1:
    #        node.syntax = syntaxname[:sep]
    #        node.word = syntaxname[sep+1:]
    #    else:
    node.syntax = syntaxname
    # read node info (incl. children)
    word = ""
    while i < len(l):
        c = l[i]
        i += 1
        if c == '(':
            if len(word) != 0:
                raise Exception(
                    "Found begin parenthese while parsing word {}:\"{}\"".
                    format(i, word))
            if len(syntaxname) == 0:
                raise Exception(
                    "Found begin parenthese while having no syntax name (implying leaf) {}".
                    format(i))
            child = node.add_child()
            i = parse_line(l, i, child)
            continue
        elif c == ')':
            if len(word) != 0:
                if node.word != None:
                    raise Exception(
                        "found word in tree but node already has word {}:\"{}\"".
                        format(i, word))
                node.word = word
                # word = ""
            else:
                if len(syntaxname) == 0:
                    raise Exception(
                        "Ending node without syntaxname {}".format(i))
            break
        elif c == ' ':
            if len(word) != 0:
                if node.parent == None:
                    if Warnings.warnRootNodeWord < 1:
                        print("WARN: found root node with word encoded. We allow as a workaround")
                    Warnings.warnRootNodeWord += 1
                    node.word = word
                    word = ""
                else:
                    raise Exception(
                        "Found space while parsing word {}:\"{}\"".format(i, word))
            continue
        elif c == '[':
            if len(word) != 0:
                raise Exception(
                    "Found square bracket while parsing word {}:\"{}\"".format(
                        i, word))
            i = read_rep(l, i, node)
            continue
        else:
            word += c
    return i


def get_tree(line, fn="get_tree"):
    if not line.startswith(" ("):
        raise Exception(fn + " line does not start with \" (\"")
    tree = Node(None)
    # print("Line is " + line)
    i = parse_line(line, 2, tree)
    if i < len(line) - 1:  # Not all of line parsed
        raise Exception(
            fn + " parsing line failed. There was more than one tree in the line. {}".
            format(i))
    l2 = output(tree)
    if l2 != line:  # Lines differ
        raise Exception(fn + " marshall and unmarshalling differs" + "\n" + line + "\n" + l2)
    if not tree.is_binary():
        raise Exception(fn + " tree is not binary")
    if not tree.has_only_words_at_leafs():
        raise Exception(fn + " tree is not properly normalized")
    return tree


# filename = "trees/train.txt"
# max_count = -1
def get_trees(file, max_count=-1):
    filename = file
    count = 0
    trees = []
    fn = "get_trees"  # function name
    Warnings.warnRootNodeWord = 0
    # aifileWrapper = ai_util.AIFileWrapper(filename)
    # aifileWrapper.__enter__()
    with ai_util.AIFileWrapper(filename) as aifileWrapper:
        for line in aifileWrapper.fd:
            line = aifileWrapper.toStrippedString(line)
            if max_count > -1 and count > max_count:
                break
            tree = get_tree(line, fn)
            if tree.is_leaf():
                print("tree is one word. Ignoring")
                continue
            trees.append(tree)
            count += 1
            if count % 2000 == 0:
                print("Extracted: ", count)
    print(fn + " done reading {}. Count={}. Roots with words count {}".format(filename, count, Warnings.warnRootNodeWord))
    return trees


def put_trees(filename, trees):
    count = 0
    fn = "put_trees "  # function name
    with io.open(filename, 'w', encoding='utf8') as f:
        for t in trees:
            treeString = output(t)
            f.write(treeString + "\n")
            count += 1
    print(fn + "done. Count={}".format(count))


def get_fraction(trees, report=False):
    count_0 = 0
    count_4 = 0
    for t in trees:
        if t.syntax == "0":
            count_0 += 1
        elif t.syntax == "4":
            count_4 += 1
        else:
            raise Exception(
                "unknown syntax encountered in tree: {}".format(t.syntax))
    res = count_4 / (count_0 + count_4)
    if report:
        print("Total count {}, sensitive {} fraction sensitive {:.4f}".format(
            count_0 + count_4, count_4, res))
    return res


def escape_sentence(l2):
    "removes parenteses and other stuff from sentence"
    l2 = l2.replace('(', '-LRB- ').replace(')', ' -RRB-')
    l2 = l2.replace('[', '-LSB- ').replace(']', ' -RSB-')
    l2 = l2.replace('{', '-LCB- ').replace('}', ' -RCB-')
    l2 = l2.replace('&', '-AMP-')
    return l2


def unescape_sentence(l2):
    "adds right parenteses and other stuff from sentence. Note we can't add correct spacing (user may use spacing in a number of different ways)"
    l2 = l2.replace('-LRB-', '(').replace('-RRB-', ')')
    l2 = l2.replace('-LSB-', '[').replace('-RSB-', ']')
    l2 = l2.replace('-LCB-', '{').replace('-RCB-', '}')
    l2 = l2.replace('-AMP-', '&')
    return l2


def count_leaf_nodes(node, count=0):
    if node == None:
        return count
    if node.is_leaf():
        return count + 1
    count = count_leaf_nodes(node.left, count)
    count = count_leaf_nodes(node.right, count)
    return count


def count_non_leaf_nodes(node, count=0):
    if node == None:
        return count
    if node.is_leaf():
        return count
    count += 1
    count = count_non_leaf_nodes(node.left, count)
    count = count_non_leaf_nodes(node.right, count)
    return count


def count_nodes(node, count=0):
    if node == None:
        return count
    count += 1
    count = count_non_leaf_nodes(node.left, count)
    count = count_non_leaf_nodes(node.right, count)
    return count


# cleaners
def cleanTreesByLength(trees, sentenceCutoffLow=5, sentenceCutoffHigh=200):
    res = []
    removedLong = 0
    removedShort = 0
    for t in trees:
        length = count_leaf_nodes(t)
        high = length
        if sentenceCutoffHigh != -1:
            high = sentenceCutoffHigh
        if length > sentenceCutoffLow and length <= high:
            res.append(t)
        elif length <= sentenceCutoffLow:
            removedShort += 1
        else:
            removedLong += 1

    print("cleanTreesByLength: removed short {}, removed long {}. Previous {} New length {}".format(removedShort, removedLong, len(trees), len(res)))
    return res


def cleanTreesByBadChars(trees):
    res = []
    ignoredSentenceCount = 0
    for t in trees:
        sentence = output_sentence(t)
        for c in sentence:
            if c < '\x1f':
                if c != '\n' and c != '\r' and c != '\f':
                    ignoredSentenceCount += 1
                    continue  # ignore this sentence
        res.append(t)
    print("cleanTreesByBadChars Cleaned: {}".format(ignoredSentenceCount))
    return res


class SentenceCounter:
    def __init__(self, trees, labelList):
        # count number of occurences of each sentence (mostly relevant for repeating sentences)
        self.labelList = labelList
        self.labelMap = {}
        for i, label in enumerate(labelList):
            self.labelMap[label] = i
        self.sentences = {}
        for t in trees:
            sentence = output_sentence(t)
            if sentence not in self.sentences:
                self.sentences[sentence] = [0 for label in labelList]
            self.sentences[sentence][self.labelMap[t.syntax]] += 1
        self.ambigousCount = 0
        self.ignoredSentenceCount = 0
        self.callCount = 0
        print("SentenceCounter initialized")

    def isSentenceAmbigous(self, sentence, sentenceLabel):
        self.callCount += 1
        countValues = 0
        for labelIndex in range(len(self.labelList)):
            if self.sentences[sentence][labelIndex] > 0:
                countValues += 1
        if countValues == 1:
            # unambigous
            return False
        else:
            # ambigous
            self.ambigousCount += 1
            maxLabelIndex = 0
            for labelIndex in range(len(self.labelList)):
                if self.sentences[sentence][labelIndex] > self.sentences[sentence][maxLabelIndex]:
                    maxLabelIndex = labelIndex
            if self.labelMap[sentenceLabel] == maxLabelIndex:
                return False
            self.ignoredSentenceCount += 1
            return True

    def cleanAmbigous(self, trees):
        res = []
        for t in trees:
            sentence = output_sentence(t)
            if not self.isSentenceAmbigous(sentence, t.syntax):
                res.append(t)
        self.reportFindings()
        return res

    def reportFindings(self):
        print("contented {} ignored {}. New size {}".format(self.ambigousCount, self.ignoredSentenceCount, self.callCount - self.ignoredSentenceCount))

