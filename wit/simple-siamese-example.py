import pandas as pd
import urllib2

from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

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

# -- Config + Init

# Siamese network example 

num_features = 1000 # Character
max_len      = 150  # Character

formatter    = KerasFormatter(num_features, max_len)

# -- Training

# Load and format data
url          = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/ak_bill_actions.csv'
raw_df       = pd.read_csv(url)
raw_df['id'] = range(raw_df.shape[0])
df           = pd.melt(raw_df, id_vars = 'id')
df.columns   = ('id', 'hash', 'obj')

pairwise_train = PairwiseData(df)
train          = pairwise_train.make_strat(neg_prop = 0.005) # Downsampling negative examples, otherwise negative set is very large

# Format for keras training
trn, val, levs = formatter.format_symmetric_with_val(train, ['obj1', 'obj2'], 'match', val_prop = .5)

# Compile and train classifier
classifier = SiameseClassifier(trn, val, levs)
classifier.fit(nb_epoch = 20)


# -- Application

# Make an artificially split dataset
sel = np.random.uniform(0, 1, df.shape[0]) > 0.5
df1 = df[sel]
df2 = df[~sel]

df1.hash = df1.hash.apply(lambda x: x + '_1')
df2.hash = df2.hash.apply(lambda x: x + '_2')

tdf = pd.concat([df1, df2])

# Predict on random pairs of entries
test = strat_pairs(tdf, n_nonmatch = 50, n_match = 50)
tst  = formatter.format_symmetric(test, ['obj1', 'obj2'], 'match')

preds         = classifier.predict(tst['x'])[:,1]
preds.shape   = (preds.shape[0], )
test['preds'] = preds[:test.shape[0]]

# Examining results
self_sims, sims = make_self_sims(test)

# Internal similarity of fields
# Note the low self similarity for actor and chamber
# This is because they are syntactically identical, and so have
# been "pushed away from themselves"
self_sims

# Column Equivalency classes
# fails on "actor" and "chamber" for aforementioned reason
# How would this be resolved?
sims[sims.sim > .9]


