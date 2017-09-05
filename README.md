# Taboo-core

To clone:

```
git clone https://bitbucket.alexandra.dk/scm/tab/taboo-core.git
```

## Notes:

Runs python3(!)

Dependencies: numpy, theano, nltk, click and flask

pip3 (or conda or apt-get or ...) install numpy, theano, nltk, click, flask

Ex:
```
sudo apt-get install python3-pip

pip3 install theano nltk flask
```

## Glove
Download glove word embeddings: (not necessary for running tests)
```
./download_glove.sh
```

This installs glove word embedding files in `../code/glove/`

## NLTK
run: `nltk.download()`, load package pickle

ex:
```
python3 -c "import nltk; nltk.download()"
```

select pickle

select Download

## Testing installation
To run tests: from taboo-core, say:

```
./run_tests.sh
```

## Training

```
OMP_NUM_THREADS=2 ipython3 functionality/train_model.py -- -traintrees 201/train_custom250_random.txt -validtrees 201/dev_custom250_random.txt -testtrees 201/test_custom250_random.txt -nx 50 -nh 100 -lr 0.01 -n_epochs 10 -glove_path ../code/glove/
```

`OMP_NUM_THREADS=2` sets the maximum numbers of threads used. We are memory intensive so best performance is with lower number of threads.

`ipython3` ipython3 or python3

`functionality/train_model.py` main code file to run (i.e. training)

`--` ipython syntax for "rest of arguments are passed on to main routine"

`-traintrees 201/train_custom250_random.txt` a set of labeled parse-trees for training

`-validtrees 201/dev_custom250_random.txt` a set of labeled parse-trees for validation (dev set)

`-testtrees 201/test_custom250_random.txt` a set of labeled parse-trees for test

`-nx 50` size of word embeddings

`-nh 100` size of hidden state

`-lr 0.01` learning rate

`-n_epochs 10` number of iterations before stopping (-1 for infinity)

`-glove_path ../code/glove/` path to location of word embeddings

There are many more options, run `train_model.py` without arguments to get the list