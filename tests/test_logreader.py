# -*- coding: utf-8 -*-
"""

Created on January 18, 2018

@author:  neerbek
"""

import unittest

import tests.RunTimer
import LogFileReader

class LogReaderTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_one(self):
        epochs = LogFileReader.readLogFile(inputfile="tests/resources/exp150.zip$exp150.log")
        self.assertEqual(199, len(epochs.loglines))
        self.assertEqual(1, epochs.loglines[0].epoch)
        self.assertEqual(96, epochs.loglines[-1].epoch)
        self.assertEqual(11, epochs.loglines[20].epoch)
        for i in range(len(epochs.loglines)):
            self.assertEqual((i + 1) * 1000, epochs.loglines[i].count)

    def test_two(self):
        epochs = LogFileReader.readLogFile(inputfile="tests/resources/testlog_001.zip$testlog_001.log")
        self.assertEqual(2, len(epochs.loglines))
        self.assertEqual(2, epochs.loglines[0].epoch)
        self.assertEqual(3, epochs.loglines[-1].epoch)
        self.assertEqual(-1, epochs.loglines[0].validationBest.epoch)
        self.assertEqual(1.234603, epochs.loglines[1].validation.cost)
        # print("cost is", epochs.loglines[1].validation.cost)
        self.assertEqual(51606, epochs.loglines[0].train.nodeCount)


if __name__ == "__main__":
    unittest.main()
