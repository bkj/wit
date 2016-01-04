import keras
import pandas as pd
import urllib2

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

num_features = 75 # Character
# max_len      = 100 # Character
max_len      = 350

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data
in_store = pd.HDFStore(
    '/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5',
    complevel = 9, 
    complib   = 'bzip2'
)
source = in_store.keys()[3]
df     = in_store[source]
in_store.close()

# Subset to frequent paths
chash = df.groupby('hash').apply(lambda x: len(x.obj.unique()))
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]
df['content'] = df.obj.apply(lambda x: BeautifulSoup(x).text.encode('utf8'))

# --
# Make all pairs

train = make_triplet_train(df, N = 600)
pd.crosstab(train.doc, train.hash)
trn, _ = formatter.format(train, ['content'], 'hash')

# Test set of all unique points
unq = df.copy()
del unq['id']
unq = unq.drop_duplicates()
awl, _ = formatter.format(unq, ['content'], 'hash')

# --
# Defining model

recurrent_size = 32
dense_size     = 5

model = Sequential()
model.add(Embedding(num_features, recurrent_size))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.add(Activation('unit_norm'))
model.compile(loss = 'triplet_euclidean', optimizer = 'adam')

# --
# Training model

# Shuffles while maintaining groups
ms = modsel(train.shape[0], N = 3)
_  = model.fit(
    trn['x'][0][ms], trn['x'][0][ms], 
    nb_epoch   = 1,
    batch_size = 3 * 250,
    shuffle    = False
)

preds = model.predict(awl['x'][0], verbose = True)

colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,1], c = colors)
plt.show()

# --
# Clustering results
#
# Could do better -- actually may want some kind of metric for "projection overlap"
from sklearn.cluster import DBSCAN
db = DBSCAN(eps = .1, min_samples = 50).fit(preds)

res         = unq.hash.groupby(db.labels_).apply(lambda x: x.value_counts()).reset_index()
res.columns = ('cluster', 'hash', 'cnt')
res         = res.sort('hash')

good_res = res[(res.cnt > 50) & (res.cluster > -1)]
good_res

sorted(res.hash.unique())
sorted(good_res.hash.unique())

eqv = list(good_res.groupby('cluster').hash.apply(lambda x: list(x)))
eqv = map(eval, np.unique(map(str, eqv)))
print_eqv(eqv, df)






