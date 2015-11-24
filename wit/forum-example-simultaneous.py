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


# -- Helpers

# Helper function for making testing data
def strat_pairs(df, n_match = 100, n_nonmatch = 10, hash_id = 'hash'):
    print 'strat_pairs -- starting'
    
    out = []
    uh  = df[hash_id].unique()
    ds  = dict([(u, df[df[hash_id] == u]) for u in uh])
    for u1 in uh:
        d1 = ds[u1]
        print 'strat_pairs :: %s' % u1
        
        for u2 in uh:
            d2  = ds[u2]
            
            cnt = n_match if (u1 == u2) else n_nonmatch
            s1  = d1.sample(cnt, replace = True).reset_index()
            s2  = d2.sample(cnt, replace = True).reset_index()
            
            # If we were using this for training, 
            # we'd want to remove these because they're non-informative
            # not_same = s1.obj != s2.obj
            # s1       = s1[not_same]
            # s2       = s2[not_same]
            
            out.append(pd.DataFrame(data = {
                "obj1"   : s1['obj'],
                "obj2"   : s2['obj'],
                
                "hash1"  : s1[hash_id],
                "hash2"  : s2[hash_id],
                
                "match"  : (s1[hash_id] == s2[hash_id]) + 0
            }))
    
    return pd.concat(out)

# Helper function for viewing aggregate similarity between fields
def make_self_sims(x):
    tmp = x.groupby(['hash1', 'hash2'])['preds']
    sims = pd.DataFrame({
        'sim' : tmp.agg(np.median),
        'cnt' : tmp.agg(len),
        'sum' : tmp.agg(sum)
    }).reset_index()
    
    sims.sim  = sims.sim.round(4)
    self_sims = sims[sims['hash1'] == sims['hash2']].sort('sim')
    return self_sims, sims

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_eqv(eqv):
    for e in eqv:
        print bcolors.WARNING + '\n --- \n'
        print e
        print '\n' + bcolors.ENDC
        for h in e:
            print bcolors.OKGREEN + h + '\t(%d rows)' % df[df.hash == h].shape[0] + bcolors.ENDC
            print df[df.hash == h].obj.head()


# -- 
# Config + Init

num_features = 500 # Modeling by Character
max_len      = 250 # Modeling by Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data

in_store = pd.HDFStore('gun_leaves_20151118_v2.h5',complevel=9, complib='bzip2')

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
