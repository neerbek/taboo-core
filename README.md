# Taboo-core

To clone:

```
git clone https://github.com/neerbek/taboo-core.git
```

## Notes:

Runs python3(!)

Dependencies: numpy, theano, nltk, click, flask, matplotlib, tkinter

pip3 (or conda or apt-get or ...) install <dependencies>

Ex:
```
sudo apt-get install python3-pip
sudo apt-get install python3-tk  # needed for tkinter

sudo -H pip3 install -r requirements.txt
```

## Glove
Download glove word embeddings: (not necessary for running tests) (requires curl to be installed)
```
./download_glove.sh
```

This installs glove word embedding files in `../code/glove/`

## NLTK
run: `nltk.download()`, load package punkt

ex:
```
python3 -m nltk.downloader punkt
```

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

## Other Commands

### run_model

```
export TREES=path-to-zip-file-with-data/trees0.zip\$test.txt
export MODEL=path-to-trained-model
export GLOVEPATH=path-to-glove

OMP_NUM_THREADS=2 ipython3 functionality/run_model.py -- -inputtrees $TREES -inputmodel $MODEL -nx 100 -nh 100 -L1_reg 0 -L2_reg 0.0001 -retain_probabilities 0.9 -batch_size 1000 -glove_path $GLOVEPATH -random_seed 1234
```

### Get Embeddings

To get a list of all embeddings generated in the final layer of a particular model

```
export TREES=path-to-zip-file-with-data/trees0.zip\$test.txt
export MODEL=path-to-trained-model
export GLOVEPATH=path-to-glove

OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees $TREES -inputmodel $MODEL -nx 100 -nh 100 -L1_reg 0 -L2_reg 0.0001 -retain_probability 0.9 -batch_size 1000 -glove_path $GLOVEPATH -random_seed 1234 -output_embeddings > output.txt
```
