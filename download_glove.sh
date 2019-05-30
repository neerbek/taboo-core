#!/bin/sh
cd ..
mkdir code
mkdir code/glove
cd code/glove/
curl -O http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
cd ../..
cd taboo-core
