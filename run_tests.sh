#!/bin/sh

#import subprocess
#args = ['python3', '-m', 'unittest', 'discover','-s','tests']
#process = subprocess.Popen(args, stdout=subprocess.PIPE)
#out, err = process.communicate()
#print(out)
python3 -m unittest discover -s tests
