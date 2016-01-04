import pandas as pd
import urllib2

from pprint import pprint
from matplotlib import pyplot as plt

from keras.layers.convolutional import Convolution1D, MaxPooling1D

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

# -- 
# Config + Init

num_features   = 500 # Character
max_len        = 250 # Character
embedding_size = 128

lstm_output_size = 64

filter_length = 3
nb_filter     = 64
pool_length   = 2

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data
in_store = pd.HDFStore('/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5',complevel=9, complib='bzip2')
source   = in_store.keys()[1]
df       = in_store[source]
in_store.close()

# Subset to frequent paths
chash = df.hash.value_counts()
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]

# -- 
# Training
pairwise_train = PairwiseData(df)
train          = pairwise_train.make_strat(n_pos = 100, neg_prop = .01) # Downsampling negative examples, otherwise negative set is very large

# Format for keras training
trn, levs = formatter.format_symmetric(train, ['obj1', 'obj2'], 'match')

val = pairwise_train.make_strat(n_pos = 100, neg_prop = .01) # Downsampling negative examples, otherwise negative set is very large
val = formatter.format(val, ['obj1'], 'hash1')

classifier = SiameseClassifier(trn, levs)
classifier.fit(nb_epoch = 15, batch_size = 128)


# --
# Convolutional model

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

# set parameters:
batch_size     = 32
embedding_dims = 100
nb_filter      = 250
filter_length  = 5
hidden_dims    = 250
nb_epoch       = 3

model = Sequential()

model.add(Embedding(num_features, embedding_dims, input_length=max_len))
model.add(Dropout(0.5))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(
    nb_filter        = nb_filter,
    filter_length    = filter_length,
    border_mode      = "valid",
    activation       = "relu",
    subsample_length = 1)
)

# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer, so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(trn['y'].shape[1]))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")

model.fit(
    trn['x'][0], trn['y'], 
    batch_size      = batch_size, 
    nb_epoch        = nb_epoch, 
    show_accuracy   = True
)

preds = model.predict(val['x'][0], verbose = True)
pd.crosstab(val['y'].argmax(1), preds.argmax(1))

# --

string_model = StringClassifier(trn, formatter.levs)
string_model.fit()

preds2 = string_model.predict(val['x'][0], verbose = True)
np.mean(val['y'].argmax(1) == preds2.argmax(1))
pd.crosstab(val['y'].argmax(1), preds2.argmax(1))



