import pandas as pd
import urllib2

from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

# --
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

# --
# Siamese network example 

num_features = 1000 # Character
max_len      = 150  # Character

formatter    = KerasFormatter(num_features, max_len)

# Load data
url          = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/mps_tanzania.csv'
raw_df       = pd.read_csv(url)
raw_df['id'] = range(raw_df.shape[0])
df           = pd.melt(raw_df, id_vars = 'id')
df.columns   = ('id', 'hash', 'obj')

pairwise_train = PairwiseData(df)
train          = pairwise_train.make_dstrat(prop = 0.05)

# Format for keras training
trn, val, levs = formatter.format_symmetric_with_val(train, ['obj1', 'obj2'], 'match', val_prop = .5)

# Compile and train classifier
classifier = SiameseClassifier(trn, val, levs)
classifier.fit()

# -- Apply to an artificially split dataset
sel = np.random.uniform(0, 1, df.shape[0]) > 0.5
df1 = df[sel]
df2 = df[~sel]

df1.hash = df1.hash.apply(lambda x: x + '_1')
df2.hash = df2.hash.apply(lambda x: x + '_2')

tdf = pd.concat([df1, df2])

test = strat_pairs(tdf, n_nonmatch = 25, n_match = 25)
tst  = formatter.format_symmetric(test, ['obj1', 'obj2'], 'match')

preds = classifier.predict(tst['x'])

pd.crosstab(tst['y'].argmax(1), preds.argmax(1))

# -- Dev

preds         = preds[:,1]
preds.shape   = (preds.shape[0], )
test['preds'] = preds[:test.shape[0]]

self_sims, sims = make_self_sims(test)
self_sims

# East to resolve fields
sims[sims.sim > 0.9][['hash1', 'hash2', 'sim']]

# Can't resolve these -- too similar maybe
pprint(sorted(list(set(sims.hash1.unique()).difference(set(sims.hash1[sims.sim > .9].unique())))))


# --
# How do I deal with categorical variables?



