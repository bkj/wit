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
for k in in_store.keys():
    df[k] = in_store[k]
    df[k]['source'] = k

df = pd.concat(df.values())
in_store.close()

# Post cleaning
df['shash'] = df.source.apply(lambda x: md5(x).hexdigest()[0:5])
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

train     = make_triplet_train(df, N = 600)
trn, levs = formatter.format(train, ['obj'], 'hash')

classifier = TripletClassifier(trn, levs)
classifier.fit(batch_size = 250, nb_epoch = 3)



# --
# Clustering results
#
# Need to use a better algorithm
#
# Could do better -- actually may want some kind of metric for "projection overlap"
from sklearn.cluster import DBSCAN
db = DBSCAN(eps = .1, min_samples = 50).fit(preds)

res         = unq.hash.groupby(db.labels_).apply(lambda x: x.value_counts()).reset_index()
res.columns = ('cluster', 'hash', 'cnt')
res         = res.sort('hash')

good_res = res[(res.cnt > 100) & (res.cluster > -1)].reset_index()
good_res

eqv = list(good_res.groupby('cluster').hash.apply(lambda x: list(x)))
eqv = map(eval, np.unique(map(str, eqv)))
print_eqv(eqv, df, path = 'src')

# --
# This actually does much better, I think

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation

nbrs = NearestNeighbors(n_neighbors = 24).fit(preds)
_, indices = nbrs.kneighbors(preds)

h         = np.array(unq.hash)
z         = pd.DataFrame(indices).groupby(h).apply(lambda x: pd.melt(x).value).reset_index()
z['hash'] = h[np.array(z.value)]
z         = z[['level_0', 'hash']]
z.columns = ('source', 'target')

sims = pd.crosstab(z.source, z.target)
af   = AffinityPropagation(affinity = 'precomputed').fit_predict(np.log(1 + sims))

sims = sims[af.argsort()]
sims = sims.T[af.argsort()].T

eqv = pd.crosstab(z.source, z.target).index.groupby(af).values()
print_eqv(eqv, df, path = 'src')
