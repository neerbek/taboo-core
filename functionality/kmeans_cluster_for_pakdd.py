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
/media/neerbek/446A4CA06A4C911A/Users/neerbek/jan/AIProjectsData/taboo-core/output/output_embeddings.txt
# arglist = "kmeans_cluster_for_pakdd.py -inputfile /media/neerbek/446A4CA06A4C911A/Users/neerbek/jan/AIProjectsData/taboo-core/output/output_embeddings.txt".split(" ")
# arglist = "kmeans_cluster_for_pakdd.py -inputfile ../../taboo-core/output/output_embeddings.txt".split(" ")
# arglist = "kmeans_cluster_for_pakdd.py -inputfile ../output_embeddings.txt".split(" ")
# arglist = "kmeans_cluster_for_pakdd.py -inputfile ../output_embeddingsEpoch0.txt".split(" ")
# arglist = "kmeans_cluster_for_pakdd.py -inputfile ../output_embeddingsEpoch233.txt".split(" ")
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

# c = 35
# kmeans = KMeans(n_clusters=c, random_state=rng).fit(a)

# ## find all sentences (index) which belong to 'label' cluster
def print_cluster(label, lines, kmeans, max_count=100, rng=rng):
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

def getCloseNeighborSentences(index=2, lines=lines, a=a, max_count=100, sentencePrefix=20):
    v = a[index, :]
    dist = euclid_distance_numpy_matrix_vector(a, v)
    sortIndex = numpy.argsort(dist)
    res = []
    print("clostest neigbors to:", load_trees.output_sentence(lines[index].tree))
    for i in range(min(max_count, len(lines))):
        line = lines[sortIndex[i]]
        sentence = load_trees.output_sentence(line.tree)
        res.append(sentence[:sentencePrefix])
    return res

def findIndex(searchTerm, lines=lines, max_count=100):
    count = 0
    index = -1
    for i in range(len(lines)):
        sentence = load_trees.output_sentence(lines[i].tree)
        if sentence.find(searchTerm) != -1:
            print(i, sentence)
            index = i
            count += 1
            if count >= max_count:
                break
    return index

def string2Index(sent, lines):
    res = []
    for s in sent:
        i = findIndex(s, lines, max_count=1)
        res.append(i)
    return res


# print_cluster(label=0, lines=lines, kmeans=kmeans, max_count=10, rng=rng)

# goodbyes
# index = 8916
# index = 8945
# index = 16878
# index = 17101
# index = 17855
# index = 1018  # nice
index = findIndex("sincerely", lines, max_count=1)
goodbye_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=20)
goodbye_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=-1)
goodbye_sent = [' Yours sincerely, En', ' Yours sincerely, En', ' IRREVOCABLE STANDBY', ' IRREVOCABLE STANDBY', ' Sorry for the delay', ' Signature of Compan', ' DESCRIPTION OF THE ', ' Subject: shipper im', ' Shall mean the amou', ' OK for me.', ' Please let me know ', ' Please send me a no', ' Sincerely, George R', ' ASSIGNMENT OF THE A', ' Notices shall be in', ' Kind regards, Melan', ': -LRB- 212 -RRB- 34', ' Regards, Andy *****', ' Purpose of Confiden', ' GBP Fixed Rate Day.', ' Enron Wholesale Ser', ' Holmes and Garrison', ' Enron Industrial Ma', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' EXECUTED effective ', ' Please respond to <', ' Yes, Gloal agrees.', ' PURPOSE AND PROCEDU', ' Please see example ', ' Termination of Agre', ' VP -AMP- AGC.', ' VP -AMP- AGC.', ' Shall have the mean', ' Please do not respo', ' Please do not respo', ' MASTER POWER PURCHA', ' Extensions of the G', ' How is my SDI direc', ' Limitation of Remed', ' Limitation of Remed']
# goodbye_indexes = getCloseNeighbors(index, lines, a, max_count=200)
goodbye_indexes = string2Index(goodbye_sent, lines)
for i, s in enumerate(goodbye_sent):
    print(i, s)
# dates
# index = 6819
index = findIndex("May-02", lines, max_count=1)
# date_indexes = getCloseNeighbors(index, lines, a, max_count=200)
date_sent = getCloseNeighborSentences(index, lines, a, max_count=50)
date_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=-1)
date_sent = [' Feb-02.', ' Oct-03.', ' Apr-01.', ' Voith-Eckerle.', ' Jun-99.', ' Aerovent.', ' Aug-04.', ' Jul-02.', ' Dec-99.', ' CGV10-34.', ' Magnalloy.', ' Aug-03.', ' Dec-00.', ' Jul-01.', ' EnronEntityName.', ' Feb-04.', ' Setoff.', ' OOrangUtan.', ' P-AMP-I.', ' Ladish.', ' Jul-01.', ' Sep-01.', ' 8.3.1.', ' Oct-00.', ' BOFMCAM2.', ' P-AMP-I.', ' Nov-04.', ' MPPA.', ' EcOutlook.', ' Letterhead/Logo.', ' Bank/Enron.', ' Enron/Alberta.', ' Jul-01.', ' ARTICLE2.', ' Dec-02.', ' Mar-02.', ' 58986.', ' ARTICLE3.', ' MPPA.', ' Feb-01.', ' BOFMCAM2.', ' Chromalox.', ' MPPA.', ' Feb-02.', ' May-02.', ' Jerryco.', ' MPPA.', ' MPPA.', ' Aug-04.', ' Mar-01.']
date_indexes = string2Index(date_sent, lines)
for i, s in enumerate(date_sent):
    print(i, s)

# names
# index = 5516
index = findIndex("Stacy", lines, max_count=1)
# name_indexes = getCloseNeighbors(index, lines, a, max_count=200)
name_sent = getCloseNeighborSentences(index, lines, a, max_count=50)
name_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=-1)
name_sent = [' Stacy Dickson.', ' Martha Braddy.', ' James Westgate.', ' Floating Amount Det', ' GEA Rainey.', ' Citibank ISDA.', ' Michelle Blaine.', ' Katherine Corbally.', ' Gerald Nemec.', ' Sara Shackleton.', ' Sara Shackleton.', ' Frank Grabowski.', ' Call 4,429.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Co. ENA.', ' Allen Bradly.', ' Harry Collins.', ' Market Disruption.', ' 12.02 Notices.', ' TW received $ 125,4', ' -LRB- p -RRB- Illeg', ' Senior Counsel, ENA', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' EXHIBIT A. ENRON CO', ' FW: Vince Kaminskis', ' Laurie Mayer.', ' CIBC means Canadian', ' -LRB- f -RRB- Set-o', ' -LRB- iv -RRB- Thre', ' -LRB- iv -RRB- Thre', ' -LRB- iv -RRB- Thre', ' -LRB- iv -RRB- Thre', ' -LRB- iv -RRB- Thre', ' -LRB- iv -RRB- Thre']
name_indexes = string2Index(name_sent, lines)

for i, s in enumerate(name_sent):
    print(i, s)
# sensitive - ok
# print_cluster(label=6, lines=lines, kmeans=kmeans, max_count=40, rng=rng)

# none-sensitive - ok
# print_cluster(label=18, lines=lines, kmeans=kmeans, max_count=15, rng=rng)

# index = 14304
index = findIndex("cools", lines, max_count=1)
# oilngas_indexes = getCloseNeighbors(index, lines, a, max_count=200)
oilngas_sent = getCloseNeighborSentences(index, lines, a, max_count=50)
oilngas_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=-1)
oilngas_sent = [' The oil flows throu', ' Where this Transact', ' Where this Transact', " `` Taxes '' means a", ' In addition, by sup', ' The chargers will b', ' Probability 20 99.4', ' The LPC and HPC sec', ' Project Cost Projec', ' A schedule of drawi', ' Hours worked by Sel', ' No Performance Liqu', ' The Company request', ' Historically, El Ni', ' In the absence of F', ' question the correc', ' Develop debt and eq', ' Series - Series Dan', ' -LSB- NOTE: THE REM', ' Put means an Option', ' Put means an Option', ' Put means an Option', ' Put means an Option', ' a release of the Di', ' 15.10 Notwithstandi', ' First Union Securit', ' Additions and delet', ' In the event of a c', ' WHEREAS, -LSB- Sell', ' The weather and the', ' Additional $ 50 mai', " `` Liens '' means a", ' The inner wall pane', ' Actual weather patt', ' Prior to emission t', ' Any deficiencies in', ' The probe also incl', ' NOx water cleanline', ' The cost of constru', ' If Buyer and Seller', ' In accordance with ', ' ** 1 -LRB- G -RRB -', ' Each Option Compone', ' If such period is a', ' The outer wall pane', ' -LRB- c -RRB- The G', ' Also shown on these', ' > > I have combined', ' The maximum per occ', ' By: Name: Title: EX']
oilngas_indexes = string2Index(oilngas_sent, lines)
# sensitive - not so good
# index = 8670
# getCloseNeighbors(index, lines, a, max_count=80)

for i, s in enumerate(oilngas_sent):
    print(i, s)


# index = 11282
# getCloseNeighbors(index, lines, a, max_count=30)

# sensitive, we go with this, but we can't reproduce the nice list from cluster 6...
# index = 15155
index = findIndex("1998 was", lines, max_count=1)
sensitive_sent = getCloseNeighborSentences(index, lines, a, max_count=50)
sensitive_sent = getCloseNeighborSentences(index, lines, a, max_count=50, sentencePrefix=-1)
sensitive_sent = [' 1998 was chosen as ', ' For tax purposes, E', ' It is a several hun', ' It is our understan', ' One thing that need', ' At the time, we did', ' For example, crude ', ' We wanted to know i', ' Third, once the dea', ' The scheme is as fo', ' In the event there ', ' We have archived qu', ' When a deal is ente', ' When a deal is ente', ' As before, we chose', ' Although there is a', ' It is possible to t', ' It is possible to t', ' I believe that, as ', ' The asset may be an', ' He was able to not ', ' He was able to not ', ' As with the case of', ' Since there are sev', ' Third, we wanted to', ' It is our understan', ' Asset cash flows, i', ' Asset cash flows, i', ' We started out by d', ' We started out by d', ' The team was able t', ' The team was able t', ' Although we did not', ' Furthermore, there ', ' We focused our effo', ' We focused our effo', ' Based upon this inf', ' We have produced a ', ' There were many dif', ' `` Acceptable Suret', ' The confirmations a', ' EVENTS OF CHANGE AN', ' Merrideth Eggleston', ' There is a chance t', ' However, there is n', ' Most deals are mode', ' Most deals are mode', ' The terms `` this A', ' In late 1999, they ', ' In late 1999, they ']
sensitive_indexes = string2Index(sensitive_sent, lines)
# sensitive_indexes = getCloseNeighbors(index, lines, a, max_count=200)

for i, s in enumerate(sensitive_sent):
    print(i, s)

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
    # elif i == 17937:
    #     labels["unknown"] = pylab.scatter(d['x-tsne'][i], d['y-tsne'][i], marker='o', c='y', s=size, lw=0)

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

