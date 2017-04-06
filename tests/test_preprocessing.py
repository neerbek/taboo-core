# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:44:13 2017

@author: neerbek
"""


import os
import unittest
import datetime

import server_enron_helper
import EnronDocument

class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.tick = datetime.datetime.now()

    def tearDown(self):
        self.tock = datetime.datetime.now()
        diff = self.tock - self.tick
        print("Time used in test ("+ os.path.basename(__file__)+")", self.id().split('.')[-1], (diff.total_seconds()), "sec")

    #modified copy of server_enron_helper.get_trees

    def test_nonascii(self):
        inputfile="3.317398.KFV32XJXBEP5TEF42WHM3J1JO0BHARTVB.1.txt"
        enronLabel = EnronDocument.EnronLabel(problem="201", fileid=inputfile, strata=1000, relevance="0")
        enronText = EnronDocument.EnronText(enronLabel, inputfile)
        enronTexts = [enronText]
        enronTexts = server_enron_helper.load_text(enronTexts)
        text = enronTexts[0].text
        for c in ['#','\x0b','\x0e','\x0f','\x10','\x11','\x12','\x13','\x14','\x15','\x16','\x17','\x18','\x19','\x1A','\x1B','\x1C','\x1D','\x1E','\x1F']:
            self.assertEqual(-1, text.find(c), "Expected doc to be cleaned. Found: {} at index {}".format(c, text.find(c)))


if __name__ == "__main__":
    unittest.main()
