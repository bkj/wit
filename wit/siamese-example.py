import pandas as pd
import urllib2

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

# --
# Helper function for making testing data
def strat_pairs(df, n_match = 100, n_nonmatch = 10, hash_id = 'hash'):
    print 'strat_pairs :: starting'
    
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

# Load data
url        = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/mps_tanzania.csv'
df         = pd.read_csv(url)
df['id']   = range(df.shape[0])
df         = pd.melt(df, id_vars = 'id')
df.columns = ('id', 'hash', 'obj')

pairwise_train = PairwiseData(df)
train_data     = pairwise_train.make_dstrat(prop = 0.05)

# Format for keras training
formatter        = KerasFormatter(num_features, max_len)
train, val, levs = formatter.format_symmetric_with_val(train_data, ['obj1', 'obj2'], 'match', val_prop = .5)

# Compile and train classifier
classifier = SiameseClassifier(train, val, levs)
classifier.fit()

test_data = strat_pairs(df, n_nonmatch = 25, n_match = 25)
test      = formatter.format_symmetric(test_data, ['obj1', 'obj2'], 'match')

preds = classifier.predict(test['x'])

pd.crosstab(test['y'].argmax(1), preds.argmax(1))

# -- Dev

preds             = preds[:,1]
preds.shape       = (preds.shape[0], )
test_data['preds'] = preds[:test_data.shape[0]]

self_sims, sims = make_self_sims(test_data)




