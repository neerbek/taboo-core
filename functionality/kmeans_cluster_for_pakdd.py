# -*- coding: utf-8 -*-
"""

Created on November 20, 2017

@author:  neerbek
"""
import sys
from sklearn.cluster import KMeans
import numpy
from numpy.random import RandomState
import time
from sklearn.manifold import TSNE
import pylab


import ai_util
import rnn_model.rnn
import rnn_model.learn
import rnn_model.FlatTrainer
import confusion_matrix
# import embedding
import similarity.load_trees as load_trees
# import kmeans_cluster_util as kutil

inputfile = "../../taboo-core/output_embeddings.txt"

trainParam = rnn_model.FlatTrainer.TrainParam()
trainParam.retain_probability = 0.9
trainParam.batchSize = 500
randomSeed = 7485

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()


def syntax():
    print("""syntax: kmeans_cluster_for_projector.py
-inputfile <filename> | -randomSeed <int> |
    [-h | --help | -?]

-inputfile is a list of final sentence embeddings in the format of run_model_verbose.py
-randomSeed initialize the random number generator
""")
    sys.exit()


arglist = sys.argv
# arglist = "kmeans_cluster_for_pakdd.py -inputfile ../../taboo-core/output/output_embeddings.txt".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-randomSeed':
        randomSeed = int(arg)
    else:
        # expected option with no argument
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i


lines = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)
a = confusion_matrix.get_embedding_matrix(lines, normalize=True)
confusion_matrix.verify_matrix_normalized(a)

rng = RandomState(randomSeed)

c = 35
kmeans = KMeans(n_clusters=c, random_state=rng).fit(a)

# ## find all sentences (index) which belong to 'label' cluster
def print_cluster(label=2, lines=lines, kmeans=kmeans, max_count=100, rng=rng):
    idx = []  # list of indexes which are in cluster=label
    perm = rng.permutation(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):  # permutate order of indexes (for when only printing some)
        index = perm[i]
        if kmeans.labels_[index] == label:
            idx.append(index)
    print("number of sentences in cluster:", len(idx))
    # output max_count first elements in 'label' cluster
    for i in range(min(max_count, len(idx))):
        line = lines[idx[i]]
        sentence = load_trees.output_sentence(line.tree)
        print(i, idx[i], sentence)

def euclid_distance_numpy_matrix_vector(m1, v2):
    """get euclidean distance between"""
    d1 = m1 - v2
    d1 = d1 * d1  # squared
    s = numpy.sum(d1, axis=1)  # summed
    s = numpy.sqrt(s)
    return s  # 1 column of numbers

def getCloseNeighbors(index=2, lines=lines, a=a, max_count=100):
    v = a[index, :]
    dist = euclid_distance_numpy_matrix_vector(a, v)
    sortIndex = numpy.argsort(dist)
    res = []
    print("clostest neigbors to:", load_trees.output_sentence(lines[index].tree))
    for i in range(min(max_count, len(lines))):
        line = lines[sortIndex[i]]
        sentence = load_trees.output_sentence(line.tree)
        d = dist[sortIndex[i]]
        print(sortIndex[i], d, sentence)
        res.append(sortIndex[i])
    return res

def findIndex(searchTerm, lines=lines, max_count=100):
    count = 0
    for i in range(len(lines)):
        sentence = load_trees.output_sentence(lines[i].tree)
        if sentence.find(searchTerm) != -1:
            print(i, sentence)
            count += 1
            if count > max_count:
                break


# print_cluster(label=32, lines=lines, kmeans=kmeans, max_count=10, rng=rng)

# goodbyes
# index = 8916
# index = 8945
# index = 16878
# index = 17101
# index = 17855
index = 1018  # nice
goodbye_indexes = getCloseNeighbors(index, lines, a, max_count=200)

# dates
index = 6819
date_indexes = getCloseNeighbors(index, lines, a, max_count=200)

# names
index = 5516
name_indexes = getCloseNeighbors(index, lines, a, max_count=200)


# sensitive - ok
# print_cluster(label=6, lines=lines, kmeans=kmeans, max_count=40, rng=rng)

# none-sensitive - ok
# print_cluster(label=18, lines=lines, kmeans=kmeans, max_count=15, rng=rng)

index = 14304
oilngas_indexes = getCloseNeighbors(index, lines, a, max_count=200)

# sensitive - not so good
# index = 8670
# getCloseNeighbors(index, lines, a, max_count=80)

# index = 11282
# getCloseNeighbors(index, lines, a, max_count=30)

# sensitive, we go with this, but we can't reproduce the nice list from cluster 6...
index = 15155
sensitive_indexes = getCloseNeighbors(index, lines, a, max_count=200)

# for i in range(35):
#     print("==", i, "===============")
#     print_cluster(label=i, lines=lines, kmeans=kmeans, max_count=5, rng=rng)

# print_cluster(label=26, lines=lines, kmeans=kmeans, max_count=15, rng=rng)

# findIndex("VaR statistic", lines, max_count=40)


# ratio = kutil.get_cluster_sen_ratios_impl(a, lines, kmeans)
# ratio = numpy.array(ratio)
# indexes = numpy.argsort(ratio)  # cluster_id to sort_index, asc

# i = 0; print(indexes[i], ratio[indexes[i]])
# i = 34; print(indexes[i], ratio[indexes[i]])
# i = 33; print(indexes[i], ratio[indexes[i]])
# i = 32; print(indexes[i], ratio[indexes[i]])
# i = 31; print(indexes[i], ratio[indexes[i]])
# i = 21; print(indexes[i], ratio[indexes[i]])

# print_cluster(label=18, lines=lines, kmeans=kmeans, max_count=15, rng=rng)

# index = 8670
# res = kmeans.predict(a[index, :].reshape(1, 100))

# for i in range(len(indexes)):
#     print(i, indexes[i])
# print(indexes)

# # generate unknown word emb for comparesion
# rng2 = RandomState(1234)
# nx = 100
# unknown = rng2.uniform(-1, 1, size=nx)
# unknown = numpy.array(unknown).reshape(1, 100)
# mag = numpy.max(unknown, axis=1)
# unknown = unknown / mag.reshape(len(mag), 1)  # first divide by max. max*max might be inf
# mag = numpy.sum(unknown * unknown, axis=1)
# for i in range(len(mag)):
#     if numpy.isinf(mag[i]):
#         raise Exception("magitude is inf for: {}".format(i))
# mag = numpy.sqrt(mag)
# unknown = unknown / mag.reshape(len(mag), 1)  # unit vectors
# nice visualization
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

# a = numpy.concatenate((a, unknown), axis=0)
# a.shape

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)  # 600 iter?
tsne_results = tsne.fit_transform(a)
print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

d = {}
d['x-tsne'] = tsne_results[:, 0]
d['y-tsne'] = tsne_results[:, 1]

print(len(name_indexes), len(date_indexes), len(oilngas_indexes), len(sensitive_indexes), len(goodbye_indexes))
 
confusion_matrix.new_graph("x", "y")
# pylab.plot(d['x-tsne'], d['y-tsne'], 'g:', label='\emph{tp}')
labels = {}  # save point for legend
# colors = ['g', 'm', 'k', 'r']
size = 25
for i in range(a.shape[0]):
    if i % 1000 == 0:
        print("done with", i)
    if i % 5 == 0:  # subsample plot points
        pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='o', c='k', s=size, lw=0)
for i in range(a.shape[0]):
    if i % 1000 == 0:
        print("done with", i)
    if i in name_indexes:
        labels["name"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='+', c='g', s=size)
    elif i in date_indexes:
        labels["date"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='+', c='r', s=size)
    elif i in goodbye_indexes:
        labels["goodbye"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='+', c='m', s=size)
    elif i in sensitive_indexes:
        labels["sensitive"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='o', c='r', s=size, lw=0)
    elif i in oilngas_indexes:
        labels["oilngas"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='o', c='g', s=size, lw=0)
    elif i == 17937:
        labels["unknown"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='o', c='y', s=size, lw=0)

# ### with unknown
# pylab.legend((labels["date"], labels["name"], labels["goodbye"], labels["sensitive"], labels["oilngas"], labels["unknown"]),
#              ("dates", "names", "goodbyes", "prepay", "oil\&gas", "unknown"),
#              scatterpoints=1)  # loc='lower left', ncol=3, fontsize=8
# ### without unknown
pylab.legend((labels["date"], labels["name"], labels["goodbye"], labels["sensitive"], labels["oilngas"]),
             ("dates", "names", "goodbyes", "prepay", "oil\&gas"),
             scatterpoints=1)  # loc='lower left', ncol=3, fontsize=8
# pylab.savefig("tsne_sentence_emb" + ".eps")
pylab.show()

