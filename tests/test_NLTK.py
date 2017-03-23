# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 07:00:23 2017

@author: neerbek
"""

import os
import unittest

#os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')
#os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')

import server_enron_helper
import server_rnn_helper
import rnn_enron

DEBUG_PRINT = False
rnn_enron.MAX_SENTENCE_LENGTH=80
rnn_enron.DEBUG_PRINT_VERBOSE = DEBUG_PRINT
rnn_enron.DEBUG_PRINT = DEBUG_PRINT


def get_id(filepath):
    return os.path.basename(filepath)[:-4]
    

class ParserTest(unittest.TestCase):
    #modified copy of server_enron_helper.get_trees
    def get_trees_from_doc(self, d, expected_count=-1):
        text = d.text
        label = d.enron_label.relevance
        sentences = server_rnn_helper.get_indexed_sentences(text)
        if (expected_count!=-1):
            self.assertEqual(expected_count, len(sentences), "Expected sentences count was wrong")
        else: 
            print("number of sentences: ", len(sentences))
        if DEBUG_PRINT: 
            for s in sentences:
                print("***SENTENCE: " + s.sentence)
        parserStatistics = rnn_enron.ParserStatistics()
        ttrees = server_rnn_helper.get_nltk_trees(0, sentences, parserStatistics)
        #rnn_enron.initializeTrees(ttrees, state.LT)
        if ttrees is None:
            self.assertTrue(False, "document was empty")
        for t in ttrees:
            t.replace_nodenames(label)
        return ttrees

    def test_indexed_sentences(self):
        doc2 = server_enron_helper.load_labeled_documents('tests/resources/enron_labels_test.txt', os.path.join(os.getcwd(), 'tests/resources/enron_data_test') , "201")
        self.assertTrue(len(doc2)==4, "Expected 4 docs in data dir")
        docmap = {}
        for d in doc2:
            d_id = get_id(d.filepath)
            docmap[d_id] = d
        print(docmap)
        fileids = [ "3.1155290.OFOJWQIRREEAX32BFOU15QLS531X5FNKA",  
        "3.1042627.PHWXPNGVEWJDVZE4OHDV2Z0GNLRJVO5SA",  
        "3.920913.OZVSEO5HGBVW3JO2V4APMH0CX0CBUQMXA",  
        "3.922767.IBI0MS320MRSHVJDNPA4IDIH00ZB2E24A.1" ]
        for d_id in fileids:
            self.assertIn(d_id, docmap)
#        t = self.get_trees_from_doc(doc2[0], 237)
#        self.assertIsNotNone(t, "expected tree")
#        t = self.get_trees_from_doc(doc2[1], 22)
#        self.assertIsNotNone(t, "expected tree")
        d = docmap[fileids[2]]
        t = self.get_trees_from_doc(d, 13)
        self.assertIsNotNone(t, "expected tree")
        d = docmap[fileids[1]]
        t = self.get_trees_from_doc(d, 1)
        self.assertIsNotNone(t, "expected tree")
        self.assertEqual(d.filepath, os.path.join(os.getcwd(), 'tests/resources/enron_data_test', '3.1042627.PHWXPNGVEWJDVZE4OHDV2Z0GNLRJVO5SA.txt'))
        self.assertEqual(d.text.strip(), "Li Doyle")
        d = docmap[fileids[0]]
        t = self.get_trees_from_doc(d, 24)
        self.assertIsNotNone(t, "expected tree")
        
    def test_header_line_break(self):
        s = """Multilateral Transactions
In addition, the Act creates a broad exemption for any agreement, contract or
transaction in commodities."""
        s = server_enron_helper.clean_text(s, look_for_header=False)
        sentences = server_rnn_helper.get_indexed_sentences(s)
        self.assertEqual(2, len(sentences), "Expected sentences count was wrong")

    def test_subject_line_break1(self):
        s = """Subject: line1
        line2
        BogusHeader: foo"""
        s = server_enron_helper.clean_text(s, look_for_header=True)
        sentences = server_rnn_helper.get_indexed_sentences(s)
        self.assertEqual(1, len(sentences), "Expected sentences count was wrong")
        self.assertEqual("line1\nline2", sentences[0].sentence, "Expected sentence was wrong")

    def test_subject_line_break2(self):
        s = """Subject: line1
        line2
        """
        s = server_enron_helper.clean_text(s)
        sentences = server_rnn_helper.get_indexed_sentences(s)
        self.assertEqual(1, len(sentences), "Expected sentences count was wrong")
        self.assertEqual("line1\nline2\n", sentences[0].sentence, "Expected sentence was wrong")
            
    def test_clean_doc_not_mail(self):
        #from edrm-enron-v2_allen-p_xml.zip/text_000/3.824725.EQ1SIWOIQCJEGJAXSR3JXBZWMH552WQEB.1.txt
        s ="""Success Real Estate Company
8855 Merlin Court
Houston, TX 77055
May 16, 2000
Central Insurance Agency, Inc.
Jeanette Peterson
6000 N. Lamar
Austin, TX 78761-5427

Dear Ms. Peterson:

This letter is in response to the recommendations made regarding the property at  808 W. Kingsbury, Seguin, Texas.  


A written policy has been implshall serve as authorization for Gary Allen to perform electrical maintenance on the Stagecoach Apartments.  He is an employee of Success Real Estate Company, Ltd. which owns the complex.
Thank you for your help.  Please call with questions.  (713) 853-7041. 
Success Real Estate Company, Ltd.
By:  Phillip Allen Investment Company, L.L.C.,
        its General Partner

_______________________
By:  Phillip Keith Allen
"""
        s = server_enron_helper.clean_text(s, look_for_header=True)
        sentences = server_rnn_helper.get_indexed_sentences(s)
        self.assertEqual(8, len(sentences), "Expected sentences count was wrong")
        
if __name__ == "__main__":
    unittest.main()
