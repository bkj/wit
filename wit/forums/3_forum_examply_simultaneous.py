import pandas as pd
import urllib2

from pprint import pprint
from matplotlib import pyplot as plt

from bs4 import BeautifulSoup
from hashlib import md5

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering

# -- 
# Config + Init

num_features = 500 # Modeling by Character
max_len      = 250 # Modeling by Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data

in_store = pd.HDFStore('/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5',complevel=9, complib='bzip2')

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
df['obj']   = df.src.apply(lambda x: BeautifulSoup(x).text.encode('utf-8'))

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

# Compile and train classifier
classifier = SiameseClassifier(trn, levs)
classifier.fit(nb_epoch = 15, batch_size = 128)

# Save classifier architecture + weights
model_name = 'sim_v101'
open(model_name + '_arch.json', 'w').write(classifier.model.to_json())
classifier.model.save_weights(model_name + '_wght.h5', overwrite = True)

# -- 
# Large scale application


test = strat_pairs(df, n_nonmatch = 25, n_match = 25)

tst, levs = formatter.format_symmetric(test, ['obj1', 'obj2'], 'match')

preds       = classifier.predict(tst['x'])[:,1]
preds.shape = (preds.shape[0], )
assert(preds.shape[0] == (test.shape[0] * 2))
test['preds'] = preds[:test.shape[0]]

# Clustering fields to make equivalency classes
self_sims, sims = make_self_sims(test)

simmat = sims.pivot(index = 'hash1', columns = 'hash2', values = 'sim')
simmat = (simmat + simmat.T) / 2

model  = AffinityPropagation(damping = .55, affinity='precomputed')
order  = model.fit_predict(simmat)
dm     = simmat[np.argsort(order)]
dm     = (dm.T[np.argsort(order)]).T
pcolor(dm)
colorbar()
show()

eqv = [list(simmat.columns[order == c]) for c in sorted(np.unique(order))]
print_eqv(eqv)


# >>

# --
# Notes
#
# Parameterization of models is so far totally unknown.  Could try to look
# for some performance optimizations.

# Thresholding self_sims means that you're  constraining the diameter of the 
# cluster.  This isn't necessarily realistic -- different clustering may have
# different diameters.  After each epoch, we could run some kind of clustering
# step on the features, and determine how many "known" misclassifications there 
# are in the training data.  When the rate of change flattens, we can stop the training
