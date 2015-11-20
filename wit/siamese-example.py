import pandas as pd
from wit import *

# --
# Siamese network example 

num_features = 1000
max_len      = 50

# Load some data and make pairwise comparisons
inpath   = '/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5'
in_store = pd.HDFStore(inpath,complevel=9, complib='bzip2')
source   = in_store.keys()[1]
df       = in_store[source]
in_store.close()

train_data    = PairwiseData(df)
train_data_ds = train_data.make_dstrat()

# Format for keras training
formatter        = KerasFormatter(num_features, max_len)
train, val, levs = formatter.format_symmetric_with_val(train_data_ds, ['obj1', 'obj2'], 'match')

# Compile and train classifier
classifier = SiameseClassifier(train, val, levs)
classifier.fit()

# Testing data uses this ugly function for training
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
            
            # Remove instances where objs are identical -- they aren't informative
            not_same = s1.obj != s2.obj
            s1       = s1[not_same]
            s2       = s2[not_same]
            
            out.append(pd.DataFrame(data = {
                "obj1"   : s1['obj'],
                "obj2"   : s2['obj'],
                
                "hash1"  : s1[hash_id],
                "hash2"  : s2[hash_id],
                
                "match"  : (s1[hash_id] == s2[hash_id]) + 0
            }))
    
    return pd.concat(out)


testdata = strat_pairs(df, n_nonmatch = 25, n_match = 25)
test     = formatter.format_symmetric(testdata, ['obj1', 'obj2'], 'match')

preds = classifier.predict(test['x'])

pd.crosstab(test['y'].argmax(1), preds.argmax(1))

