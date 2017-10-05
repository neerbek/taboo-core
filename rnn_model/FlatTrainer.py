import numpy
from numpy.random import RandomState
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from datetime import datetime

import rnn_model.rnn

class TrainParam:
    def __init__(self):
        self.X = None
        self.Y = None
        self.valX = None
        self.valY = None
        self.learner = None   # instance from learn
        self.batchSize = None
        self.validBatchSize = 0
        self.L1param = 0
        self.L2param = 0

class Layer:
    def __init__(self, nOut):
        self.nIn = None
        self.x = None
        self.nOut = nOut
        self.params = []
        self.regularizedParams = []

    def getL1(self):
        return 0

    def getL2(self):
        return 0

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.nIn = nIn
        self.x = x
        return

    def getPrediction(self):
        return T.sum(self.x)

class RegressionLayer:
    def __init__(self, nOut):
        self.nOut = nOut
        self.nIn = None
        self.x = None
        self.W = None
        self.b = None
        self.params = []

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.nIn = nIn
        self.x = x
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (self.nIn + self.nOut)),
                high=numpy.sqrt(6. / (self.nIn + self.nOut)),
                size=(self.nIn, self.nOut)
            ),
            dtype=theano.config.floatX
        )
        # self.W = theano.shared(value=W_values, name='W_reg', borrow=True)
        self.W = theano.shared(value=W_values, name="W_regression_{}".format(layerNumber), borrow=False)
        # initialize the biases b as a vector of n_out 0s
        b_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (self.nOut)),
                high=numpy.sqrt(6. / (self.nOut)),
                size=(self.nOut,)
            ),
            dtype=theano.config.floatX
        )
        # self.b = theano.shared(value=b_values, name='b_reg', borrow=True)
        self.b = theano.shared(value=b_values, name="b_regression_{}".format(layerNumber), borrow=False)
        self.params = [self.W, self.b]
        self.regularizedParams = [self.W]

    def getPrediction(self):
        return T.nnet.softmax(T.dot(self.x, self.W) + self.b)

class ReluLayer:
    def __init__(self, nOut):
        self.nOut = nOut
        self.nIn = None
        self.x = None
        self.W = None
        self.b = None
        self.params = []
        self.regularizedParams = []

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.nIn = nIn
        self.x = x
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (self.nIn + self.nOut)),
                high=numpy.sqrt(6. / (self.nIn + self.nOut)),
                size=(self.nIn, self.nOut)
            ),
            dtype=theano.config.floatX
        )
        # self.W = theano.shared(value=W_values, name='W_reg', borrow=True)
        self.W = theano.shared(value=W_values, name="W_reluLayer_{}".format(layerNumber), borrow=False)
        # initialize the biases b as a vector of n_out 0s
        b_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (self.nOut)),
                high=numpy.sqrt(6. / (self.nOut)),
                size=(self.nOut,)
            ),
            dtype=theano.config.floatX
        )
        # self.b = theano.shared(value=b_values, name='b_reg', borrow=True)
        self.b = theano.shared(value=b_values, name="b_reluLayer_{}".format(layerNumber), borrow=False)
        self.params = [self.W, self.b]
        self.regularizedParams = [self.W]

    def getPrediction(self):
        # TODO: add dropout
        return T.nnet.relu(T.dot(self.x, self.W) + self.b)


class DropoutLayer:
    def __init__(self, z, innerLayer):
        self.z = z
        self.nOut = innerLayer.nOut
        self.params = []
        self.regularizedParams = []
        self.rng = None
        self.nIn = None
        self.innerLayer = innerLayer

    def setInputSize(self, nIn, x, layerNumber, rng):
        self.rng = rng
        self.nIn = nIn
        self.innerLayer.setInputSize(nIn, x, layerNumber, rng)
        self.params = self.innerLayer.params
        self.regularizedParams = self.innerLayer.regularizedParams

    def getPrediction(self):
        return self.z * self.innerLayer.getPrediction()

class RNNContainer:
    def __init__(self, nIn, rng=RandomState(1234)):
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.z = T.matrix('z', dtype=theano.config.floatX)    # for dropout
        self.layers = []
        self.nIn = nIn
        self.nOut = None
        self.rng = rng

    def addLayer(self, layer):
        if len(self.layers) == 0:
            layer.setInputSize(self.nIn, self.x, 0, self.rng)
        else:
            previousLayer = self.layers[-1]
            layer.setInputSize(previousLayer.nOut, previousLayer.getPrediction(), len(self.layers), self.rng)
        self.nOut = layer.nOut
        self.layers.append(layer)

    def getParams(self):
        params = []
        for l in self.layers:
            params.extend(l.params)
        return params

    def getOutputPrediction(self):
        return self.layers[-1].getPrediction()

    def getL1(self):
        l1 = 0
        for l in self.layers:
            for p in l.regularizedParams:
                l1 += abs(p).sum()
        return l1

    def getL2(self):
        l2 = 0
        for l in self.layers:
            for p in l.regularizedParams:
                l2 += (p ** 2).sum()
        return l2

    def load(self, filename):
        return rnn_model.rnn.layeredLoad(model=self, filename=filename)

    def save(self, filename, epoch, acc):
        rnn_model.rnn.layeredSave(model=self, filename=filename, epoch=epoch, acc=acc)

class FlatPerformanceMeasurer:
    def __init__(self, epoch=-1):
        self.epoch = epoch
        self.accuracy = 0
        self.cost = 0
        self.countZeros = 0
        self.confusionMatrix = [0, 0, 0, 0]

    def report(self, msg=""):
        print(msg + " total accuracy {:.4f} % ({:.4f} %) cost {:.6f}".format(self.accuracy * 100., self.countZeros * 100., self.cost * 1.0))

class ModelEvaluator:
    def __init__(self, model, trainParam, withDropout=False):
        self.model = model
        self.trainParam = trainParam
        inputs = [model.x, model.y]
        if withDropout:
            inputs = [model.x, model.y, model.z]
        self.costFunction = theano.function(
            inputs=inputs,
            outputs=self.cost()
        )
        self.accuracyFunction = theano.function(
            inputs=inputs,
            outputs=self.accuracy()
        )
        self.confusionMatrixFunction = theano.function(
            inputs=inputs,
            outputs=self.confusionMatrix()
        )

    def cost(self):  # RMS cost, theano
        pred = self.model.getOutputPrediction()
        err = pred - self.model.y
        c = T.mean(0.5 * ((err) ** 2))
        c += self.trainParam.L1param * self.model.getL1()
        c += self.trainParam.L2param * self.model.getL2()
        return c

    def getCost(self, X, Y):
        return self.costFunction(X, Y)

    def accuracy(self):  # unweigthed average accuracy, theano
        pred = self.model.getOutputPrediction()
        pred = T.argmax(pred, axis=1)
        y_simple = T.argmax(self.model.y, axis=1)  # assumes input is a matrix
        # check if y has same dimension of y_pred
        if y_simple.ndim != pred.ndim:
            raise TypeError(
                'y_simple should have the same shape as pred',
                ('y_simple', y_simple.type, 'pred', pred.type)
            )
        # check if y is of the correct datatype
        if y_simple.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.eq(pred, y_simple))
        else:
            raise NotImplementedError()

    def getAccuracy(self, X, Y):
        return self.accuracyFunction(X, Y)

    def confusionMatrix(self):
        """ count of true and false preds
        """
        y_pred = T.argmax(self.model.getOutputPrediction(), axis=1)
        y_simple = T.argmax(self.model.y, axis=1)
        first_row = T.ones_like(self.model.y[0, :])
        TRUE_CONSTANT = T.sum(first_row)
        FALSE_CONSTANT = 1
        all_t = T.eq(TRUE_CONSTANT, y_simple)
        all_f = T.eq(FALSE_CONSTANT, y_simple)  # since we _know_ that labels are either TRUE or FALSE
        count_ones = T.ones_like(y_simple)
        # isOk = ifelse(T.sum(all_t) + T.sum(all_f) == T.sum(count_ones), True, False)
        # if T.neq(T.sum(all_t) + T.sum(all_f), T.sum(count_ones)):
        #      raise Exception("Expected only true-false predictions. (t={}, f={}), count={}, size={}".format(TRUE_CONSTANT, FALSE_CONSTANT, T.sum(all_t) + T.sum(all_f), T.sum(count_ones)))
        pred_t = T.eq(TRUE_CONSTANT, y_pred)
        pred_f = T.neq(TRUE_CONSTANT, y_pred)  # since we can predict more than 0/4
        tp = T.sum(T.eq(1, all_t * pred_t))  # all_t*pred_t=1 if tp
        fp = T.sum(T.eq(1, all_f * pred_t))
        tn = T.sum(T.eq(1, all_f * pred_f))
        fn = T.sum(T.eq(1, all_t * pred_f))
        return (tp, fp, tn, fn)

    def getConfusionMatrix(self, X, Y):
        return self.confusionMatrixFunction(X, Y)

def measure(X, Y, batchSize, modelEvaluator):
    perf = FlatPerformanceMeasurer()
    n = X.shape[0]
    nBatches = int(numpy.ceil(n / batchSize))
    for i in range(nBatches):
        xIn = X[i * batchSize: (i + 1) * batchSize]
        yIn = Y[i * batchSize: (i + 1) * batchSize]
        # z_val = retain_probability * numpy.ones(shape=(x_val.shape[0], rnn.n_hidden), dtype=theano.config.floatX)
        currentSize = xIn.shape[0]
        perf.accuracy += modelEvaluator.getAccuracy(xIn, yIn) * currentSize  # append fraction accuracy
        perf.cost += modelEvaluator.getCost(xIn, yIn) * currentSize        # append fraction cost
        perf.countZeros += numpy.sum(yIn[:, 0])
        a = modelEvaluator.getConfusionMatrix(xIn, yIn)
        for i in range(len(perf.confusionMatrix)):
            perf.confusionMatrix[i] += a[i]  # a is list of arrays (theano is weird)
    perf.accuracy = perf.accuracy / n
    perf.countZeros = perf.countZeros / n
    perf.cost = perf.cost / n
    return perf

def train(trainParam, rnnContainer, file_prefix="save", n_epochs=1, rng=RandomState(1234), epoch=0, validationFrequency=1, trainReportFrequency=1):
    it = 0
    n = trainParam.X.shape[0]  # number of examples
    repSize = trainParam.X.shape[1]  # size of representations
    resSize = trainParam.Y.shape[1]  # size of answers

    nTrainBatches = int(numpy.ceil(n / trainParam.batchSize))
    if trainParam.validBatchSize == 0:
        trainParam.validBatchSize = trainParam.valX.shape[0]
    nValidBatches = int(numpy.ceil(trainParam.valX.shape[0] / trainParam.validBatchSize))

    modelEvaluator = ModelEvaluator(rnnContainer, trainParam)

    updates = trainParam.learner.getUpdates(rnnContainer.getParams(), modelEvaluator.cost)
    updateKeys = [k for k in updates.keys()]  # updateKeys are needed for updating learning

    outputs = [modelEvaluator.accuracy, modelEvaluator.cost] + [updates[k] for k in updateKeys]

    train = theano.function(inputs=[rnnContainer.X, rnnContainer.Y],
                            outputs=outputs)

    performanceMeasurerBest = FlatPerformanceMeasurer()
    performanceMeasurerBest.runningEpoch = -1
    performanceMeasurer = FlatPerformanceMeasurer(-1)

    while (n_epochs == -1 or epoch < n_epochs):
        perm = rng.permutation(n)
        X = [trainParam.X[i, :].reshape(1, repSize) for i in perm]
        Y = [trainParam.Y[i, :].reshape(1, resSize) for i in perm]
        X = numpy.concatenate(X)
        Y = numpy.concatenate(Y)

        epoch += 1
        trainCost = 0
        trainAcc = 0
        trainCount = 0
        for minibatchIndex in range(nTrainBatches):
            startIndex = minibatchIndex * trainParam.batchSize
            xIn = X[startIndex: startIndex + trainParam.batchSize, :]
            yIn = Y[startIndex: startIndex + trainParam.batchSize, :]

            currentSize = xIn.shape[0]
            if currentSize == 0:
                print("empty train iteration")
                continue
            # TODO: add dropout
            # z_in = rng.binomial(n=1, size=(current_size, rep_size), p=retain_probability)
            # z_in = z_in.astype(dtype=theano.config.floatX)
            values = train(xIn, yIn)
            trainAcc += values[0] * currentSize
            trainCost += values[1] * currentSize
            trainCount += currentSize
            for index, param in enumerate(updateKeys):
                param.set_value(values[index + 2])
            # Timers.calltheanotimer.end()
            it += 1
            if it % trainReportFrequency == 0:
                minibatchZeros = yIn.shape[0] - numpy.sum(yIn[:, 0])
                minibatchAcc = values[0]  # validate_model(x_val, y_val, z_val)
                print("{} Epoch {}. On train set : Node count {}, avg cost {:.6f}, avg acc {:.4f}%".format(
                    datetime.now().strftime('%d%m%y %H:%M'), epoch, trainCount, trainCost / trainCount, trainAcc / trainCount * 100.))
                print("minibatch {}/{}, On train set: batch acc {:.4f} %  ({:.4f} %)".format(minibatchIndex + 1, nTrainBatches, minibatchAcc * 100.0, minibatchZeros * 100.0))

            if it % validationFrequency == 0:
                performanceMeasurer = measure(trainParam.valX, trainParam.valY, nValidBatches, modelEvaluator)
                performanceMeasurer.report(msg="{} Epoch {}. On validation set: Best ({}, {:.6f}, {:.4f}%). Current: ".format(
                    datetime.now().strftime('%d%m%y %H:%M'), epoch, performanceMeasurerBest.epoch, performanceMeasurerBest.cost * 1.0, performanceMeasurerBest.accuracy * 100.))
                cm = performanceMeasurer.confusionMatrix
                if trainParam.valX.shape[0] != cm[0] + cm[1] + cm[2] + cm[3]:
                    raise Exception("Expected total_node_count to be equal to sum", trainParam.valX.shape[0], cm[0] + cm[1] + cm[2] + cm[3])
                print("Confusion Matrix validation (tp,fp,tn,fn)", cm[0], cm[1], cm[2], cm[3])
                if performanceMeasurerBest.accuracy < performanceMeasurer.accuracy:
                    saveBest(model=rnnContainer, filePrefix=file_prefix, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                    performanceMeasurerBest = performanceMeasurer
                    performanceMeasurerBest.runningEpoch = epoch
                else:
                    if performanceMeasurerBest.runningEpoch + 1 < epoch:
                        saveRunning(model=rnnContainer, filePrefix=file_prefix, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)
                        performanceMeasurerBest.runningEpoch = epoch
    # loop done, save current model
    saveRunning(model=rnnContainer, filePrefix=file_prefix, epoch=epoch, performanceMeasurer=performanceMeasurer, performanceMeasurerBest=performanceMeasurerBest)


def saveRunning(model, filePrefix, epoch, performanceMeasurer, performanceMeasurerBest):
    filename = "{}_running.txt".format(filePrefix)
    save(model, filename, epoch, performanceMeasurer, performanceMeasurerBest)

def saveBest(model, filePrefix, epoch, performanceMeasurer, performanceMeasurerBest):
    filename = "{}_best.txt".format(filePrefix)
    save(model, filename, epoch, performanceMeasurer, performanceMeasurerBest)

def save(model, filename, epoch, performanceMeasurer, performanceMeasurerBest):
    print("Saving flat model. Previous {};{:.4f}. New {};{:.4f}".format(performanceMeasurerBest.epoch, performanceMeasurerBest.accuracy, performanceMeasurer.epoch, performanceMeasurer.accuracy))
    print("Saving as " + filename)
    model.save(filename=filename, epoch=epoch, acc=performanceMeasurer.accuracy)

