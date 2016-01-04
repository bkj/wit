import pandas as pd
import urllib2

from pprint import pprint
from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

# -- 
# Config + Init

num_features = 500 # Character
max_len      = 250 # Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data
in_store = pd.HDFStore('/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5',complevel=9, complib='bzip2')
source   = in_store.keys()[1]
df       = in_store[source]
in_store.close()

# Subset to frequent paths
chash = df.hash.value_counts()
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]

# -- 
# Training
pairwise_train = PairwiseData(df)
train          = pairwise_train.make_strat(n_pos = 50, neg_prop = .001) # Downsampling negative examples, otherwise negative set is very large

# Format for keras training
trn, levs = formatter.format_symmetric(train, ['obj1', 'obj2'], 'match')

# Compile and train classifier
classifier = SiameseClassifier(trn, levs)
classifier.fit(nb_epoch = 15, batch_size = 128)

preds         = classifier.predict(trn['x'])[:,1]
preds.shape   = (preds.shape[0], )
assert(preds.shape[0] == (train.shape[0] * 2))
train['preds'] = preds[:train.shape[0]]

self_sims, sims = make_self_sims(train)
self_sims


# -- Application

# Predict on random pairs of entries
test = strat_pairs(df, n_nonmatch = 20, n_match = 20)
tst, levs = formatter.format_symmetric(test, ['obj1', 'obj2'], 'match')

preds         = classifier.predict(tst['x'])[:,1]
preds.shape   = (preds.shape[0], )
assert(preds.shape[0] == (test.shape[0] * 2))
test['preds'] = preds[:test.shape[0]]

# Examining results
self_sims, sims = make_self_sims(test)

# Column Equivalency classes
sims[sims.sim > .8]

equivs, uequivs = make_equiv(test, THRESH = .9)
eqv             = uequivs.values()
print_eqv(eqv)


# --
import theano

tst['x'][1] = tst['x'][0]
preds = classifier.predict(tst['x'])

glo1 = theano.function([model.layers[0].layers[0].layers[0].input], 
                                       model.layers[0].layers[0].layers[2].get_output(train=False))
glo2 = theano.function([model.layers[0].layers[1].layers[0].input], 
                                       model.layers[0].layers[1].layers[2].get_output(train=False))

lo1 = glo1(tst['x'][0][0:10])
lo2 = glo2(tst['x'][0][0:10])

sum(lo1[0] * lo1[0])


