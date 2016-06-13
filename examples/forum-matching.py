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
sys.path.append('..')
from wit import *
from mmd import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

np.set_printoptions(linewidth=100)

from pylab import pcolor, show, colorbar
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

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
# Make training pairs

train     = make_triplet_train(df, N = 600)
trn, levs = formatter.format(train, ['obj'], 'hash')

# --
# Train model

classifier = TripletClassifier(trn, levs)
classifier.fit(batch_size = 250, nb_epoch = 3)

# --
# Predict

unq = df.copy()
del unq['id']
unq = unq.groupby(list(unq.columns)).size().reset_index()
unq.rename(columns = {0 : 'freq'}, inplace = True)

awl, _ = formatter.format(unq, ['obj'], 'hash')
preds  = classifier.predict(awl['x'][0], verbose = True)

# --
# Clustering -- either of these methods are viable

# --
# Using cosine MMD

labs = awl['y'].argmax(1)

out = np.zeros( (len(levs), len(levs)) )
for i in range(len(levs)):
    print i
    a = preds[labs == i]
    for j in range(i + 1, len(levs)):
        b = preds[labs == j]
        out[i, j] = mmd(a, b)


out_orig = out
out = out + out.T

sims = np.exp(-out)
sims[sims > 1] = 1
af   = AffinityPropagation(damping = 0.5, affinity = 'precomputed').fit_predict(sims)

sims = sims[af.argsort()]
sims = sims.T[af.argsort()].T

eqv = map(lambda x: [levs[i] for i in x], pd.DataFrame(levs).groupby(af).groups.values())
print_eqv(eqv, df, path = 'obj')

# --
# Using nearest neighbors


nbrs = NearestNeighbors(n_neighbors = 24).fit(preds)
_, indices = nbrs.kneighbors(preds)

h         = np.array(unq.hash)
z         = pd.DataFrame(indices).groupby(h).apply(lambda x: pd.melt(x).value).reset_index()
z['hash'] = h[np.array(z.value)]
z         = z[['level_0', 'hash']]
z.columns = ('source', 'target')

sims = pd.crosstab(z.source, z.target)
sims = sims + sims.transpose()
# sims = sims.apply(lambda x: x / sum(x), 1)
af   = AffinityPropagation(affinity = 'precomputed').fit_predict(np.log(1 + sims))

sims = sims[af.argsort()]
sims = sims.T[af.argsort()].T

pcolor(sims)
colorbar()
show()

eqv = pd.crosstab(z.source, z.target).index.groupby(af).values()
print_eqv(eqv, df, path = 'obj')
