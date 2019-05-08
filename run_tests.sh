#!/bin/sh

#from python
#import subprocess
#args = ['python3', '-m', 'unittest', 'discover','-s','tests']
#process = subprocess.Popen(args, stdout=subprocess.PIPE)
#out, err = process.communicate()
#print(out)
#
#run single test: OMP_NUM_THREADS=2 python3 -m unittest tests.test_NLTK.ParserTest.test_whitespace2
export TABOO_CORE_NO_LATEX=1
echo Tests requiring Latex has been disabled
OMP_NUM_THREADS=2 python3 -m unittest discover -s tests
