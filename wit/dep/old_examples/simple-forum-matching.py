# --
# Load deps

import keras
import pandas as pd
import urllib2

from hashlib import md5
from bs4 import BeautifulSoup
from pprint import pprint
from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

np.set_printoptions(linewidth=100)

# -- 
# Config + Init

num_features = 75  # Character
max_len      = 350 # Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data

path     = '/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5'
in_store = pd.HDFStore(path,complevel=9, complib='bzip2')

df = {}
for k in ['/www.remingtonsociety.com', '/marauderairrifle.com']:
    df[k]           = in_store[k]
    df[k]['origin'] = k

df = pd.concat(df.values())
in_store.close()

# Post cleaning
df['shash'] = df.origin.apply(lambda x: md5(x).hexdigest()[0:5])
df['hash']  = df.apply(lambda x: str(x['hash']) + '-' + x['shash'], 1)
df['id']    = df.apply(lambda x: str(x['id']) + '-' + x['shash'], 1)
df['src']   = df.obj
df['obj']   = df.src.apply(lambda x: BeautifulSoup(x).text.encode('ascii', 'ignore'))

# Subset to frequent paths, w/ more than 100 unique values
chash = df.groupby('hash').apply(lambda x: len(x.obj.unique()))
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]

# --
# Make all pairs

train = make_triplet_train(df, N = 600)
trn, _ = formatter.format(train, ['obj'], 'hash')

# Test set of all unique points
unq = df.copy()
del unq['id']
unq    = unq.drop_duplicates()
awl, _ = formatter.format(unq, ['obj'], 'hash')

# --
# Define model

recurrent_size = 32 # How to pick?
dense_size     = 5  # How to pick?

model = Sequential()
model.add(Embedding(num_features, recurrent_size))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.add(Activation('unit_norm'))
model.compile(loss = 'triplet_cosine', optimizer = 'adam')

# --
# Train model

# Shuffles while maintaining groups
N = 3
for _ in range(N):
    ms = modsel(train.shape[0], N = 3)
    _  = model.fit(
        trn['x'][0][ms], trn['x'][0][ms], 
        nb_epoch   = 1,
        batch_size = 3 * 250,
        shuffle    = False
    )

preds = model.predict(awl['x'][0], verbose = True)

colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,2], c = colors)
plt.show()

# --
# Clustering results

# This is not the ideal algorithm for clustering the results, 
# but it does an OK job.  
#
# In this case we're losing some of the fields
#

from sklearn.cluster import DBSCAN
db = DBSCAN(eps = .1, min_samples = 50).fit(preds)

res         = unq.hash.groupby(db.labels_).apply(lambda x: x.value_counts()).reset_index()
res.columns = ('cluster', 'hash', 'cnt')
res         = res.sort('hash')

good_res = res[(res.cnt > 100) & (res.cluster > -1)]
good_res

missing_hashes = set(res.hash.unique()).difference(set(good_res.hash.unique()))
res[res.hash.isin(missing_hashes)]

eqv = list(good_res.groupby('cluster').hash.apply(lambda x: list(x)))
eqv = map(eval, np.unique(map(str, eqv)))
print_eqv(eqv, df, path = 'src')

