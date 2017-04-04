# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 07:00:23 2017

@author: neerbek
"""

import os
import unittest
import datetime
#os.chdir('/Users/neerbek/jan/phd/DLP/paraphrase/python')
#os.chdir('/home/neerbek/jan/phd/DLP/paraphrase/python')

import server_enron_helper
import server_rnn_helper
import rnn_enron
import EnronDocument

DEBUG_PRINT = False
rnn_enron.MAX_SENTENCE_LENGTH=80
rnn_enron.DEBUG_PRINT_VERBOSE = DEBUG_PRINT
rnn_enron.DEBUG_PRINT = DEBUG_PRINT


def get_id(filepath):
    return os.path.basename(filepath)[:-4]
    

class ParserTest(unittest.TestCase):
    def setUp(self):
        self.tick = datetime.datetime.now()

    def tearDown(self):
        self.tock = datetime.datetime.now()
        diff = self.tock - self.tick
        print("Time used in test (test_NLTK)", self.id().split('.')[-1], (diff.total_seconds()), "sec")

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
        #FILE 1
        d = docmap[fileids[0]]
        sentences = server_rnn_helper.get_indexed_sentences(d.text)
        self.assertEqual(24, len(sentences), "Expected sentences count was wrong")
        #shorter text to make test run faster
        d.text = sentences[0].sentence+"\n"+sentences[3].sentence+"\n"+sentences[5].sentence+"\n"+sentences[6].sentence
        t = self.get_trees_from_doc(d, 4)
        self.assertIsNotNone(t, "expected tree")
        #FILE 2
        d = docmap[fileids[1]]
        t = self.get_trees_from_doc(d, 1)
        self.assertIsNotNone(t, "expected tree")
        self.assertEqual(d.filepath, os.path.join(os.getcwd(), 'tests/resources/enron_data_test', '3.1042627.PHWXPNGVEWJDVZE4OHDV2Z0GNLRJVO5SA.txt'))
        self.assertEqual(d.text.strip(), "Li Doyle")
        #FILE 3
        d = docmap[fileids[2]]
        sentences = server_rnn_helper.get_indexed_sentences(d.text)
        self.assertEqual(13, len(sentences), "Expected sentences count was wrong")
        #shorter text to make test run faster
        d.text = sentences[0].sentence+"\n"+sentences[1].sentence+"\n"+sentences[7].sentence+"\n"+sentences[8].sentence;
        t = self.get_trees_from_doc(d, 4)
        self.assertIsNotNone(t, "expected tree")
        #FILE 4
        d = docmap[fileids[3]]
        sentences = server_rnn_helper.get_indexed_sentences(d.text)
        self.assertEqual(318, len(sentences), "Expected sentences count was wrong")
        
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
        
        
    def test_whitespace(self):
        s ="""-LRB- B -RRB- Threshold means, with respect to a party -LRB- a -RRB- the amount set forth opposite the lowest Credit Rating for the party -LRB- or in the case of Party A, its Credit Support Provider -RRB- on the relevant date of determination; or -LRB- b -RRB- zero if on the relevant date of determination -LRB- i -RRB- the entity referred to in clause -LRB- a -RRB- above does not have a Credit Rating from either S-AMP-P or Moody's, or -LRB- ii -RRB- an Event of Default or Potential Event of Default with respect to such party has occurred and is continuing: THRESHOLD."""
        s = server_enron_helper.clean_text(s)
        indexedSentences = server_rnn_helper.get_indexed_sentences(s)
        t = server_rnn_helper.get_nltk_trees(0, indexedSentences)
        self.assertTrue(len(t)>0)

    def test_whitespace2(self):
        s ="\"Credit Rating\" means with respect to a party (or its Credit Support Provider, as the case may be) or entity, on any date of determination, the respective ratings then assigned to such party’s (or its Credit Support Provider's, as the case may be) or entity’s unsecured, senior long-term debt (not supported by third party credit enhancement) by S&P, Moody’s or the other specified rating agency or agencies."
        s = server_enron_helper.clean_text(s)
        indexedSentences = server_rnn_helper.get_indexed_sentences(s)
        t = server_rnn_helper.get_nltk_trees(0, indexedSentences)
        self.assertTrue(len(t)>0)


    def test_whitespace_from_file1(self):
        file="tests/resources/textExample1.txt"
        #small/selective copy of edrm-enron-v2_love-p_xml.zip/text_000/3.476695.KK4ZP3TKPLPOW0WULFMZGOKVSSKDHLVKB.1.txt
        enronText = EnronDocument.EnronText(0, file)        
        enronTexts = []
        enronTexts = server_enron_helper.load_text([enronText])
        self.assertEqual(0, len(enronTexts), "Expected doc to be cleaned")

    def test_whitespace_from_file2(self):
        file="tests/resources/textExample2.txt"
        #small/selective copy of edrm-enron-v2_shackleton-s_xml.zip/text_000/3.556787.HRYCGJJUVARPMD1XCNR4OWP3T1ZWV5NWA.2.txt"        
        enronText = EnronDocument.EnronText(0, file)        
        enronTexts = []
        enronTexts = server_enron_helper.load_text([enronText])
        self.assertEqual(1, len(enronTexts), "Expected doc to not be cleaned")
        text = enronText.text
        sentences = server_rnn_helper.get_indexed_sentences(text)
        self.assertEqual(5, len(sentences), "Expected sentences count was wrong")
        trees = server_rnn_helper.get_nltk_trees(0, sentences)
        self.assertEqual(5, len(trees), "Expected tree count was wrong")

    def test_quotes(self):
        s= """[1:50 1:58] Changed "is " to "and (ii) are "
[1:50 1:59] Changed "following:-" to ""
[1:51 1:59] Changed ""Address " to ""(i) Address  ...  follows: "
[1:51 1:59] Changed "Section " to "Sections "
"""
        indexedSentences = server_rnn_helper.get_indexed_sentences(s)
        t = server_rnn_helper.get_nltk_trees(0, indexedSentences)
        self.assertTrue(len(t)>0)

    def test_quotes_from_file(self):
        rnn_enron.MAX_SENTENCE_LENGTH=100
        file="tests/resources/textExample3.txt"
        #small/selective copy of edrm-enron-v2_shackleton-s_xml.zip/text_000/3.556787.HRYCGJJUVARPMD1XCNR4OWP3T1ZWV5NWA.2.txt"        
        enronText = EnronDocument.EnronText(0, file)        
        enronTexts = []
        enronTexts = server_enron_helper.load_text([enronText])
        self.assertEqual(1, len(enronTexts), "Expected doc to not be cleaned")
        text = enronText.text
        sentences = server_rnn_helper.get_indexed_sentences(text)
        self.assertEqual(1, len(sentences), "Expected sentences count was wrong")
        trees = server_rnn_helper.get_nltk_trees(0, sentences)
        self.assertEqual(1, len(trees), "Expected tree count was wrong")

        

if __name__ == "__main__":
    unittest.main()
