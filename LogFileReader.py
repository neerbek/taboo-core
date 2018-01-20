# -*- coding: utf-8 -*-
"""

Created on January 17, 2018

@author:  neerbek
"""
import ai_util
from datetime import datetime
from typing import List

class Train:
    def __init__(self, cost=0, nodeAccuracy=0, nodeCount=0):
        self.cost = cost
        self.nodeAccuracy = nodeAccuracy
        self.nodeCount = nodeCount

    def __str__(self):
        return "Train(nodeCount={}, {}, {})".format(self.nodeCount, self.cost, self.nodeAccuracy)

class Validation:
    def __init__(self, cost=0, nodeAccuracy=0, nodeZeros=0, rootAccuracy=0, rootZeros=0):
        self.cost = cost
        self.nodeAccuracy = nodeAccuracy
        self.nodeZeros = nodeZeros
        self.rootAccuracy = rootAccuracy
        self.rootZeros = rootZeros

    def __str__(self):
        return "Val({}, {}, {})".format(self.cost, self.rootAccuracy, self.nodeAccuracy)

class ValidationBest:
    def __init__(self, cost=0, rootAccuracy=0, epoch=0):
        self.cost = cost
        self.rootAccuracy = rootAccuracy
        self.epoch = epoch  # when we obtained this validation score

    def __str__(self):
        return "ValBest(epoch={}, {}, {})".format(self.epoch, self.cost, self.rootAccuracy)

class LogLine:
    def __init__(self, epoch=-1):
        self.epoch = epoch
        self.count = 0  # for count of number of processed minibatches
        self.train = None  # type: Train
        self.validation = None  # type: Validation
        self.validationBest = None  # type: ValidationBest

    def isComplete(self) -> bool:
        # i.e
        # epoch is set
        # train, validation, validations are not None
        return (self.epoch != -1 and self.train != None and self.validation != None and self.validationBest != None)

class LogLines:
    def __init__(self):
        self.loglines: List[LogLine]; self.loglines = []

    def addTrain(self, train: Train, epoch: int) -> None:
        if len(self.loglines) == 0:
            self.loglines.append(LogLine())
        if self.loglines[-1].isComplete():
            self.loglines.append(LogLine())
        self.loglines[-1].epoch = epoch
        self.loglines[-1].train = train

    def addValidation(self, validation: Validation, validationBest: ValidationBest, epoch: int) -> None:
        if len(self.loglines) == 0 or self.loglines[-1].isComplete():
            raise Exception("We are trying to add validation, but no corresponding train logline was found. Epoch={}, validation={}".format(epoch, validation))
        if self.loglines[-1].epoch != epoch:
            raise Exception("We are trying to add validation, but corresponding train logline has wrong epoch. Epoch={}, validation={}".format(epoch, validation))
        self.loglines[-1].validation = validation
        self.loglines[-1].validationBest = validationBest

    def addCount(self, count):
        if len(self.loglines) == 0:
            raise Exception("We trying to it count, but loglines are empty")
        if not self.loglines[-1].isComplete():
            raise Exception("We trying to it count, but current logline as no validation. Epoch={}, train={}".format(self.loglines[-1].epoch, self.loglines[-1].train))
        self.loglines[-1].count = count

    def checkLast(self):
        if len(self.loglines) > 0 and not self.loglines[-1].isComplete():
            print("missing validation score from last epoch, removing")
            del self.loglines[-1]


def readLogFile(inputfile: str) -> LogLines:
    count = 0
    epochs = LogLines()
    # inputfile="logs/exp150.zip$exp150.log"
    # log = ai_util.AIFileWrapper(inputfile).__enter__()
    #
    # lines = log.fd.readlines()
    # lines = [log.toStrippedString(line) for line in lines]
    # len(lines)
    # log.__exit__(None, None, None)
    # line = lines[100]
    # print(line)
    with ai_util.AIFileWrapper(inputfile) as log:
        for line in log.fd:
            line = log.toStrippedString(line)
            if count % 30000 == 0:
                print("read: {}".format(count))
            count += 1
            if len(line) == 0:
                continue
            e = line.split(' ')
            if len(e) < 17 and len(e) != 3:
                continue
            # print("line", line)
            # DDMMYY HH:MM Epoch <e>. On <dataset> set ...
            if len(e) == 3:
                if e[0] == "Saving" and e[1] == "as" and e[2].endswith(".txt"):
                    index = e[2].rfind("_")
                    if index == -1:
                        continue
                    substr = e[2][index + 1:-4]
                    if substr == "best" or substr == "running":
                        continue
                    count = int(substr)
                    epochs.addCount(count)
            elif len(e[0]) == 6 and len(e[1]) == 5 and e[2] == "Epoch" and e[4] == "On":
                # assume this is an interesting line
                if e[5] == "train":
                    epoch = int(e[3][:-1])  # remove last .
                    train = Train()
                    train.nodeCount = int(e[10][:-1])
                    train.cost = float(e[13][:-1])
                    train.nodeAccuracy = float(e[16][:-3])
                    epochs.addTrain(train, epoch=epoch)
                elif e[5] == "validation":
                    if len(e) < 24:
                        raise Exception("Line does not contain all expected information. line is " + line)
                    epoch = int(e[3][:-1])
                    validation = Validation()
                    validation.cost = float(e[20][:-1])
                    validation.rootAccuracy = float(e[23])
                    validation.nodeAccuracy = float(e[15])
                    validationBest = ValidationBest()
                    validationBest.epoch = int(e[8][1:-1])
                    validationBest.cost = float(e[9][1:-1])
                    validationBest.rootAccuracy = float(e[10][:-3])
                    epochs.addValidation(validation, validationBest, epoch=epoch)
                else:
                    raise Exception("unknown dataset at e[5]. line is: " + line)
    print("Done. read: {}".format(count))
    epochs.checkLast()
    return epochs


def logTrain(train: Train, epoch: int, dataSetName="train") -> None:
    msg = "{} Epoch {}. On {} set : Node count {}, avg cost {:.6f}, avg acc {:.4f}%"
    print(msg.format(datetime.now().strftime('%d%m%y %H:%M'), epoch, dataSetName, train.nodeCount, train.cost, train.nodeAccuracy * 100.))

def logValidation(validation: Validation, validationBest: ValidationBest, epoch: int) -> None:
    msg = "{} Epoch {}. On validation set: Best ({}, {:.6f}, {:.4f}%). Current: "
    msg = msg.format(datetime.now().strftime('%d%m%y %H:%M'), epoch, validationBest.epoch, validationBest.cost * 1.0, validationBest.rootAccuracy * 100.)
    msg2 = " total accuracy {:.4f} % ({:.4f} %) cost {:.6f}, root acc {:.4f} % ({:.4f} %)"
    msg2 = msg2.format(validation.nodeAccuracy * 100., validation.nodeZeros * 100.,
                       validation.cost * 1.0, validation.rootAccuracy * 100.,
                       validation.rootZeros * 100.)
    print(msg + msg2)


if __name__ == "__main__":
    # epochs = readLogFile(inputfile="logs/exp151.zip$exp151.log")
    epochs = readLogFile(inputfile="tests/resources/exp150.zip$exp150.log")
    print("read: ", len(epochs.loglines))
    for logline in epochs.loglines:
        print(logline.epoch, logline.count)
