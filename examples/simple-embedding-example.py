# --
# WIT :: String embedding example

import urllib2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import sys
sys.path.append('..')
from wit import *

print 'WIT :: Initializing'
num_features = 100 # Characters
max_len      = 150 # Characters
formatter    = KerasFormatter(num_features, max_len)

print 'WIT :: Loading data'
url          = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/ak_bill_actions.csv'
raw_df       = pd.read_csv(url)
sel          = np.random.choice(range(raw_df.shape[0]), 1000)
raw_df       = raw_df.iloc[sel].reset_index()
raw_df['id'] = range(raw_df.shape[0])

print 'WIT :: Subsetting to a couple of types, for illustration'
raw_df = raw_df[['id', 'bill_id', 'date', 'action']]

print 'WIT :: Melting data'
df           = pd.melt(raw_df, id_vars = 'id')
df.columns   = ('id', 'hash', 'obj')

print 'WIT :: Making training set'
train = make_triplet_train(df, N = 500)

print 'WIT :: Formatting data'
trn, levs = formatter.format(train, ['obj'], 'hash')

print 'WIT :: Compiling model'
classifier = TripletClassifier(trn, levs, opts = {"recurrent_size" : 32, "dense_size" : 2})
classifier.fit(batch_size = 50, nb_epoch = 5)

print 'WIT :: Embedding all points'
awl, _  = formatter.format(df, ['obj'], 'hash')
preds   = classifier.model.predict(awl['x'][0], verbose = True)

print 'WIT :: Plotting the embedding'
colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,1], c = colors)
plt.show()