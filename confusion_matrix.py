# -*- coding: utf-8 -*-
"""
Created on August 16 2017

@author: neerbek
"""

import io

import pylab
import scipy
import numpy

import similarity.load_trees as load_trees

class Line:
    def __init__(self, tree=None, emb=None, ground_truth=0, is_correct=0, sen_score=0):
        self.tree = tree
        self.emb = emb
        self.ground_truth = ground_truth
        self.is_correct = is_correct
        self.sen_score = sen_score


class ConfusionMatrix:
    TYPE_TP = 0
    TYPE_FP = 1
    TYPE_TN = 2
    TYPE_FN = 3
    TYPE_LIST = [TYPE_TP, TYPE_FP, TYPE_TN, TYPE_FN]

    def __init__(self, index, sentence):
        self.sentence = sentence.lower()
        self.exact_matches = 0
        self.tp = 0
        self.tp_score = []
        self.fp = 0
        self.fp_score = []
        self.tn = 0
        self.tn_score = []
        self.fn = 0
        self.fn_score = []
        self.worst_distance = 1  # distance to added neighbour which is the furthest away (i.e. worst neightbour in terms of distance)
        self.t_pos = None  # cutoff index for positive threshold
        self.t_neg = None  # cutoff index for negative threshold

    def compare(self, rhs):
        # self.sentence.equal
        if self.sentence != rhs.sentence:
            return False
        if self.exact_matches != rhs.exact_matches:
            return False
        if self.tp != rhs.tp:
            return False
        if not numpy.array_equal(self.tp_score, rhs.tp_score):
            return False
        if self.fp != rhs.fp:
            return False
        if not numpy.array_equal(self.fp_score, rhs.fp_score):
            return False
        if self.tn != rhs.tn:
            return False
        if not numpy.array_equal(self.tn_score, rhs.tn_score):
            return False
        if self.fn != rhs.fn:
            return False
        if not numpy.array_equal(self.fn_score, rhs.fn_score):
            return False
        if self.worst_distance != rhs.worst_distance:
            return False
        if self.t_pos != rhs.t_pos:
            return False
        if self.t_neg != rhs.t_neg:
            return False
        return True

    def add(self, dist, line):
        sentence = load_trees.output_sentence(line.tree)
        if self.sentence == sentence.lower():
            self.exact_matches += 1
        score = line.sen_score
        cmType = None
        if line.ground_truth == 0:
            if line.is_correct == 0:
                cmType = ConfusionMatrix.TYPE_FP
            else:
                cmType = ConfusionMatrix.TYPE_TN
        else:
            if line.is_correct == 0:
                cmType = ConfusionMatrix.TYPE_FN
            else:
                cmType = ConfusionMatrix.TYPE_TP

        if cmType == ConfusionMatrix.TYPE_TP:
            self.tp += 1
            self.tp_score.append(score)
        elif cmType == ConfusionMatrix.TYPE_FP:
            self.fp += 1
            self.fp_score.append(score)
        elif cmType == ConfusionMatrix.TYPE_TN:
            self.tn += 1
            self.tn_score.append(score)
        elif cmType == ConfusionMatrix.TYPE_FN:
            self.fn += 1
            self.fn_score.append(score)
        else:
            raise Exception("Unknown type in ConfusionMatrix.add")
        self.worst_distance = dist

    def get_score(self, cmType):
        if cmType == ConfusionMatrix.TYPE_TP:
            return self.tp_score
        elif cmType == ConfusionMatrix.TYPE_FP:
            return self.fp_score
        elif cmType == ConfusionMatrix.TYPE_TN:
            return self.tn_score
        elif cmType == ConfusionMatrix.TYPE_FN:
            return self.fn_score
        else:
            raise Exception("Unknown type in ConfusionMatrix.get_score")

    def get_avg_score(self, cmType):
        count = 0
        score = 0

        score = self.get_score(cmType)
        count = len(score)
        if count == 0:
            return 0
        s = 0
        for sc in score:
            s += sc
        return s / count

    def get_std_dev_score(self, cmType):
        avg = self.get_avg_score(cmType)
        count = 0
        score = self.get_score(cmType)
        count = len(score)

        if count == 0:
            return 0
        s = 0
        for sc in score:
            s += (avg - sc)**2
        s = s / count
        return numpy.sqrt(s)

    def report(self):
        print("(tp, fp, tn, fn) = ({}, {}, {}, {}), scores = ({:.2f} ({:.2f}), {:.2f} ({:.2f}), {:.2f} ({:.2f}), {:.2f} ({:.2f})) (exact_matches: {})".format(self.tp, self.fp, self.tn, self.fn, self.get_avg_score(ConfusionMatrix.TYPE_TP), self.get_std_dev_score(ConfusionMatrix.TYPE_TP), self.get_avg_score(ConfusionMatrix.TYPE_FP), self.get_std_dev_score(ConfusionMatrix.TYPE_FP), self.get_avg_score(ConfusionMatrix.TYPE_TN), self.get_std_dev_score(ConfusionMatrix.TYPE_TN), self.get_avg_score(ConfusionMatrix.TYPE_FN), self.get_std_dev_score(ConfusionMatrix.TYPE_FN), self.exact_matches))

class ConfusionMatrixSimpleElement:
    def __init__(self):
        self.count = 0
        self.score = 0
        self.std_dev = 0


class ConfusionMatrixSimple:
    def __init__(self, index, score, ground_truth):
        self.index = index
        self.score = score
        self.ground_truth = ground_truth
        self.scores = {}
        self.scores[ConfusionMatrix.TYPE_TP] = ConfusionMatrixSimpleElement()
        self.scores[ConfusionMatrix.TYPE_FP] = ConfusionMatrixSimpleElement()
        self.scores[ConfusionMatrix.TYPE_TN] = ConfusionMatrixSimpleElement()
        self.scores[ConfusionMatrix.TYPE_FN] = ConfusionMatrixSimpleElement()
        self.worst_distance = 1  # distance to added neighbour which is the furthest away (i.e. worst neightbour in terms of distance)

    def load_values(self, confusionMatrix):
        for t in ConfusionMatrix.TYPE_LIST:
            element = self.scores[t]
            element.count = len(confusionMatrix.get_score(t))
            element.score = confusionMatrix.get_avg_score(t)
            element.std_dev = confusionMatrix.get_std_dev_score(t)

    def is_within(self, t):
        element = self.scores[t]
        if element.count == 0:
            return False
        min_score = element.score - (2 * element.std_dev)
        max_score = element.score + (2 * element.std_dev)
        return (self.score > min_score and self.score < max_score)

    def score_dist(self, t):
        element = self.scores[t]
        if element.count == 0:
            return 1
        return abs(element.score - self.score)

    def get_standard_prediction(self):
        pred = 4
        if self.score < 0.5:
            pred = 0
        return pred

    def predict(self):
        pred = None
        # t = ConfusionMatrix.TYPE_TN
        min_score_dist = 1
        if self.score < 0.5:
            t = ConfusionMatrix.TYPE_TN
            if self.is_within(t):
                return 0
        else:
            t = ConfusionMatrix.TYPE_TP
            if self.is_within(t):
                return 4
        for t in ConfusionMatrix.TYPE_LIST:
            dist = self.score_dist(t)
            if dist < min_score_dist:
                min_score_dist = dist
                if t == ConfusionMatrix.TYPE_TP:
                    pred = 4
                elif t == ConfusionMatrix.TYPE_FP:
                    pred = 0
                elif t == ConfusionMatrix.TYPE_TN:
                    pred = 0
                else:
                    if self.score < 0.01:
                        pred = 0
                    else:
                        pred = 4
        return pred

def plot_cm(cm, index, save=False, show=False, show_normal=None, scale_equally=True, show_additive=False, scaleFactor=1):
    """Plots the the [tp,fp,tn,fn] over sensitivity score. If show_normal
is set (list of cm.get_avg_score(t), cm.get_std_dev_score(t),
len(cm.get_score(t)) for each t) then scale_equally tells if we scale
according to count (len(cm.get_score(t))) or with equal constant
scaling. Show_additive tells whether we will add the normal
distributions or just show max
    """
    # see http://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html
    fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt = 490.0  # fullwidth (maybe)
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (numpy.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    x = numpy.arange(0, 1.01, 0.01)
    y = [[] for i in x]
    colors = ['g', 'm', 'k', 'r']

    for t in cm.TYPE_LIST:
        marker = 'o'
        color = 'g'
        if t == cm.TYPE_TP:
            marker = 'o'
            color = colors[t]
        elif t == cm.TYPE_FP:
            marker = '+'
            color = colors[t]
        elif t == cm.TYPE_TN:
            marker = 'o'
            color = colors[t]
        elif t == cm.TYPE_FN:
            marker = '+'
            color = colors[t]
        else:
            raise Exception("unknown type")

        print("getting elements: {}".format(len(cm.get_score(t))))
        for s in cm.get_score(t):
            s = round(s, 2)
            i = int(s * 100)
            if i < 0:
                i = 0
            elif i > 100:
                i = 100
            l = y[i]
            e = [x[i], len(l), marker, color]  # [x, y, marker, color]
            l.append(e)

    # Plot data
    pylab.figure(1)
    pylab.clf()
    # pylab.axes([0.125, 0.2, 0.95 - 0.125, 0.95 - 0.2])
    count = 0
    points = {"og": None, "+m": None, "ok": None, "+r": None}
    for l in y:
        for e in l:
            if e[2] == 'o':
                points[e[2] + e[3]] = pylab.scatter(e[0], e[1], marker=e[2], c=e[3], s=7, lw=0)  # save a point for legend
            else:
                points[e[2] + e[3]] = pylab.scatter(e[0], e[1], marker=e[2], c=e[3], s=7)
            count += 1
    print("added {}".format(count))
    pylab.xlabel('Sensitive Score')
    pylab.ylabel('Counts')
    # print(points)
    if show_normal != None:
        y1 = [0 for i in x]
        for i in range(4):
            vals = scipy.stats.norm.pdf(x, loc=show_normal[3 * i], scale=show_normal[3 * i + 1])
            if scale_equally:
                vals = vals * 78 * scaleFactor  # scale equally
            else:
                vals = vals * show_normal[3 * i + 2] * scaleFactor / 4.7  # scale according to counts
            # vals = vals * show_normal[3 * i + 2] * 10  # scale according to counts
            for j in range(len(y1)):
                if show_additive:
                    y1[j] += vals[j]
                else:
                    y1[j] = max(y1[j], vals[j])
        print("MaxY {:.4f}".format(max(y1)))
        # pylab.plot(x, vals, colors[i] + ':', label='Normal Dist.')
        pylab.plot(x, y1, 'k:', label='Normal Dist.')
    pylab.legend((points["og"], points["+m"], points["ok"], points["+r"]),
                 ('$\emph{tp}$', '$\emph{fp}$', '$\emph{tn}$', '$\emph{fn}$'),
                 scatterpoints=1)  # loc='lower left', ncol=3, fontsize=8
    if save:
        pylab.savefig("fig_probs_s{}.eps".format(index))
    if show:
        pylab.show()

def new_graph(xlabel, ylabel):
    fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt = 490.0  # fullwidth (maybe)
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (numpy.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)
    # Plot data
    pylab.figure(1)
    pylab.clf()
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    # pylab.plot(x, tp, 'g:', label='\emph{tp}')
    # pylab.legend()
    # pylab.show()

def plot_graphs(x, tp, fp, tn, fn, name, show=False):
    new_graph(xlabel='Sensitivity Score', ylabel='Accumelated Counts')
    pylab.plot(x, tp, 'g:', label='\emph{tp}')
    pylab.plot(x, fp, '-m', label='\emph{fp}')
    pylab.plot(x, tn, 'k:', label='\emph{tn}')
    pylab.plot(x, fn, '-r', label='\emph{fn}')
    # pylab.axvline(x=0.38)
    # pylab.axvline(x=0.98)
    pylab.legend()
    if name != None:
        pylab.savefig(name + '.eps')
    if show:
        pylab.show()


def finding_cm(dist, indices, index, lines, closest_count=100):
    index_dist = dist[:, index]        # column
    index_indices = indices[:, index]  # column
    sentence = load_trees.output_sentence(lines[index].tree)
    cm = ConfusionMatrix(index, sentence)
    # i = 0
    for i in range(closest_count):
        best = index_indices[i]
        best_dist = index_dist[best]
        cm.add(best_dist, lines[best])
    return cm

def find_threshold(cm):
    x = numpy.arange(0, 1.01, 0.01)
    pos = [[0, 0] for i in x]
    neg = [[0, 0] for i in x]

    # get counts per x
    # countList \in \{pos, neg\}
    # index \in \{0,1\}
    for (t, countList, index) in [(cm.TYPE_TP, pos, 0), (cm.TYPE_FP, pos, 1), (cm.TYPE_TN, neg, 0), (cm.TYPE_FN, neg, 1)]:
        for s in cm.get_score(t):
            i = int(round(s * 100))
            i = max(0, i)    # making sure that i is a valid index
            i = min(i, 100)  #
            l = countList[i]
            l[index] += 1

    # accumelate counts
    prev = [0, 0]
    for p in pos:
        p[0] += prev[0]
        p[1] += prev[1]
        prev = p

    prev = [0, 0]
    for n in reversed(neg):
        n[0] += prev[0]
        n[1] += prev[1]
        prev = n

    t_pos = None
    for i in reversed(range(len(pos))):
        p = pos[i]
        if p[1] > p[0]:
            if t_pos is not None:
                continue
            t_pos = i
        elif p[1] == p[0]:
            pass
        else:
            t_pos = None
    cm.t_pos = t_pos

    t_neg = None
    for i in range(len(neg)):
        n = neg[i]
        if n[1] > n[0]:
            if t_neg is not None:
                continue
            t_neg = i
        elif n[1] == n[0]:
            pass
        else:
            t_neg = None
    cm.t_neg = t_neg
    return (x, pos, neg)  # for plotting

def is_all_elements_nan(numbers):
    for n in numbers:
        f = numpy.float32(n)
        if not numpy.isnan(f):
            return False
    return True

def read_embeddings(inputfile, max_line_count=-1):
    count = 0
    lines = []
    nan_indexes = []
    len_embedding = None
    with io.open(inputfile, 'r', encoding='utf8') as f:
        for line in f:
            if max_line_count > -1 and count > max_line_count:
                break
            line = line[:-1]  # strips newline. But consider: http://stackoverflow.com/questions/509446/python-reading-lines-w-o-n
            if line.startswith("Index\tTruth\tIs_Accurate\tProbSen\tSize\tSentence\tTree"):
                if not line.endswith("\tRootEmbedding"):
                    print("WARNING: No embeddings in file. Aborting")
                    break
                continue
            # print(line)
            e = line.split("\t")
            if len(e) < 2:
                continue
            emb = e[-1]
            t = e[-2]
            # print(emb)
            if not emb.startswith("["):
                print("emb has wrong begin format. Ignoring")
                continue
            if not emb.endswith("]"):
                print("emb has wrong end format. Ignoring")
                continue
            emb = emb[1:-1]  # remove "[" + "]"
            emb = emb.split(",")
            line_element = Line()
            tree = load_trees.get_tree(t)
            line_element.tree = tree
            line_element.emb = []
            line_element.ground_truth = int(e[1])
            line_element.is_correct = int(e[2])
            line_element.sen_score = numpy.float32(e[3])
            do_continue = False
            for em in emb:
                if len(em) == 0:
                    continue
                f = numpy.float32(em)  # float(em)
                if numpy.isnan(f):
                    if not is_all_elements_nan(emb):
                        raise Exception("Not all elements in line {} is nan!".format(count))
                    nan_indexes.append(count)
                    do_continue = True
                    break
                line_element.emb.append(f)
            if do_continue:
                continue  # read next line
            if len_embedding is None:
                len_embedding = len(line_element.emb)
                print("Embeddings found to have length: {}".format(len_embedding))
            else:
                if len_embedding != len(line_element.emb):
                    raise Exception("Expected all embeddings to have equal lengths {}, {}".format(len_embedding, len(line_element.emb)))
            lines.append(line_element)
            count += 1
            if count % 2000 == 0:
                print("Extracted: ", count)
    print("Done. Count={} Len={} Len(nan_indexes)={}".format(count, len(lines), len(nan_indexes)))
    return lines

def get_embedding_matrix(lines, normalize=False):
    first_emb = None
    embeddings = []
    for l in lines:
        if first_emb == None:
            first_emb = l.emb
        embeddings.append(l.emb)
    a = numpy.array(embeddings)
    print(a.shape)
    if a.shape[0] != len(lines):
        raise Exception("a is expected to contain rows of embeddings")
    for i in range(len(first_emb)):
        if not numpy.isclose(a[0, i], first_emb[i]):
            print("{}: {} and {} are different".format(i, a[i, 0], first_emb[i]))
    # ## normalize vectors
    if normalize is True:
        mag = numpy.max(a, axis=1)
        a = a / mag.reshape(len(mag), 1)  # first divide by max. max*max might be inf
        mag = numpy.sum(a * a, axis=1)
        for i in range(len(mag)):
            if numpy.isinf(mag[i]):
                raise Exception("magitude is inf for: {}".format(i))
        mag = numpy.sqrt(mag)
        a = a / mag.reshape(len(mag), 1)  # unit vectors
    return a  # a is expected to contain rows of embeddings

def verify_matrix_normalized(m, atol=0.00001):
    mag = numpy.sum(m * m, axis=1)
    for i in range(len(mag)):
        if not numpy.isclose(mag[i], 1, atol=atol):
            raise Exception("magitude is wrong for: {}".format(i))
