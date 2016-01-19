# --
# WIT :: String embedding example

import urllib2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from munkres import Munkres

import sys
sys.path.append('..')
from wit import *
from mmd import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

np.set_printoptions(linewidth = 120)

# --

print 'WIT :: Initializing'
num_features = 100 # Characters
max_len      = 150 # Characters
formatter    = KerasFormatter(num_features, max_len)

print 'WIT :: Loading data'
url    = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/ak_bill_actions.csv'
raw_df = pd.read_csv(url)

sel              = np.random.choice(range(raw_df.shape[0]), 2000)
raw_df           = raw_df.iloc[sel]
raw_df['id']     = range(raw_df.shape[0])
raw_df['schema'] = np.random.choice(('a', 'b'), 2000)

print 'WIT :: Melting data'
df         = pd.melt(raw_df, id_vars = ['id', 'schema'])
df.columns = ('id', 'schema', 'hash', 'obj')
df.id      = df.apply(lambda x: x['schema'] + '-' + str(x['id']), 1)
df.hash    = df.apply(lambda x: x['schema'] + '-' + str(x['hash']), 1)

print 'WIT :: Making training set'
train = make_triplet_train(df, N = 500)

print 'WIT :: Formatting data'
trn, levs = formatter.format(train, ['obj'], 'hash')

print 'WIT :: Compiling model'
classifier = TripletClassifier(trn, levs, opts = {"recurrent_size" : 32, "dense_size" : 4})
classifier.fit(batch_size = 50, nb_epoch = 5)

print 'WIT :: Embedding all points'
awl, _  = formatter.format(df, ['obj'], 'hash')
preds   = classifier.model.predict(awl['x'][0], verbose = True)

print 'WIT :: Plotting the embedding'
colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,1], c = colors)
plt.show()

# --
# Re-merging dataset (in progress)

labs = awl['y'].argmax(1)

out = np.zeros( (len(levs), len(levs)) )
for i in range(len(levs)):
    for j in range(len(levs)):
        out[i, j] = fast_cosine_mmd(preds[labs == i], preds[labs == j])

# --
# Linear assignment on MMD
dist         = pd.DataFrame(out.copy())
dist.columns = dist.index = levs
dist         = dist[filter(lambda i: 'a-' in i, dist.index)].loc[filter(lambda i: 'a-' not in i, dist.index)]

mmap = Munkres().compute(np.array(dist))
mmap = [ (dist.index[x[0]], dist.columns[x[1]]) for x in mmap ]
mmap = [ (x[0], x[1], dist[x[1]].loc[x[0]]) for x in mmap ]
pd.DataFrame(mmap)

# --
# Linear assignment on MMD
# In this case unnecessary step, but maybe useful in general

ddist         = pd.DataFrame(squareform(pdist(out.copy())))
ddist.columns = ddist.index = levs
ddist         = ddist[filter(lambda i: 'a-' in i, ddist.index)].loc[filter(lambda i: 'a-' not in i, ddist.index)]

mmap_d = Munkres().compute(np.array(ddist))
mmap_d = [ (ddist.index[x[0]], ddist.columns[x[1]]) for x in mmap_d ]
mmap_d = [ (x[0], x[1], ddist[x[1]].loc[x[0]]) for x in mmap_d ]
pd.DataFrame(mmap_d)

ddist[['a-actor', 'a-chamber']]
