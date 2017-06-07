#!/bin/sh

#from python
#import subprocess
#args = ['python3', '-m', 'unittest', 'discover','-s','tests']
#process = subprocess.Popen(args, stdout=subprocess.PIPE)
#out, err = process.communicate()
#print(out)
#
#run single test: OMP_NUM_THREADS=2 python3 -m unittest tests.test_NLTK.ParserTest.test_whitespace2
OMP_NUM_THREADS=2 python3 -m unittest discover -s tests
