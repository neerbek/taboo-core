#!/bin/sh
cd ..
mkdir code
mkdir code/glove
cd core/glove/
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../..
cd taboo-core