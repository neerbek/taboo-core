# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:44:13 2017

@author: neerbek
"""


import unittest

import server_enron_helper
import EnronDocument

import RunTimer

class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.timer = RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    #modified copy of server_enron_helper.get_trees

    def test_nonascii(self):
        inputfile="tests/resources/texts/3.317398.KFV32XJXBEP5TEF42WHM3J1JO0BHARTVB.1.txt"
        enronLabel = EnronDocument.EnronLabel(problem="201", fileid=inputfile, strata=1000, relevance="0")
        enronText = EnronDocument.EnronText(enronLabel, inputfile)
        enronTexts = [enronText]
        enronTexts = server_enron_helper.load_text(enronTexts)
        text = enronTexts[0].text
        for c in ['#','\x0b','\x0e','\x0f','\x10','\x11','\x12','\x13','\x14','\x15','\x16','\x17','\x18','\x19','\x1A','\x1B','\x1C','\x1D','\x1E','\x1F']:
            self.assertEqual(-1, text.find(c), "Expected doc to be cleaned. Found: {} at index {}".format(c, text.find(c)))

    def test_removedquotes(self):
        inputfile="tests/resources/texts/3.311758.AWACZMQ5U2ABLZMO0RFJOR5FHKAG5OI1B.txt"
        enronLabel = EnronDocument.EnronLabel(problem="201", fileid=inputfile, strata=1000, relevance="0")
        enronText = EnronDocument.EnronText(enronLabel, inputfile)
        enronTexts = [enronText]
        enronTexts = server_enron_helper.load_text(enronTexts)
        text = enronTexts[0].text
        for c in ['"']:
            self.assertEqual(-1, text.find(c), "Expected doc to be cleaned. Found: {} at index {}".format(c, text.find(c)))


if __name__ == "__main__":
    unittest.main()
