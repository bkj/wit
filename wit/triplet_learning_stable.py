import pandas as pd
import urllib2

from pprint import pprint
from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

# -- 
# Config + Init

num_features = 75  # Character
max_len      = 40 # Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data
in_store = pd.HDFStore(
    '/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5',
    complevel = 9, 
    complib   = 'bzip2'
)
source = in_store.keys()[1]
df     = in_store[source]
in_store.close()

# Subset to frequent paths
chash = df.hash.value_counts()
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]

# def all_neg_pairs(x):
#     out = []
#     for i, r1 in x.iterrows():
#         for j, r2 in x.iterrows():
#             if i != j:
#                 out += [r1.to_dict(), r2.to_dict()]
    
#     return pd.DataFrame(out)


# df.head(100).groupby('id').apply(all_neg_pairs)

# --
# Make all pairs

def make_train(df, N = 200):
    out     = []
    uhash   = df.hash.unique()
    
    counter = 0
    for uh in uhash:
        print uh
        pos  = df[df.hash == uh].sample(N * 2, replace = True)
        
        # neg = df[(df.hash != uh) & df.id.isin(pos.id.unique())]
        neg = df[(df.hash != uh) & df.id.isin(pos.id.unique())].sample(N, replace = True)
        
        pos['doc']  = uh
        neg['doc'] = uh
        
        for i in range(N):
            anc_ = pos.iloc[i].to_dict()
            pos_ = pos.iloc[N + i].to_dict()
            
            # neg_ = neg[neg.id == anc_['id']].sample(1).iloc[0].to_dict()
            neg_ = neg.iloc[i].to_dict()
            
            anc_['ex'] = counter
            pos_['ex'] = counter
            neg_['ex'] = counter
            
            out += [ anc_, pos_, neg_ ]
            
            counter += 1
    
    return pd.DataFrame(out)

# T = time()
# train = make_train(df.head(500), N = 400)
# time() - T

# subhash = ['5fd24', '3122c', 'e5316', '6a138']
hcounts = df.hash.value_counts()
subhash = np.array(hcounts[hcounts > 1200].index)
sub     = df[df.hash.isin(subhash)]
tmp     = sub.id.value_counts()
sub     = sub[sub.id.isin(np.array(tmp[tmp > 1].index))]

# sub = df

train = make_train(sub, N = 1200)

pd.crosstab(train.doc, train.hash)

trn, _ = formatter.format(train, ['obj'], 'hash')
awl, _ = formatter.format(sub, ['obj'], 'hash')

# --

recurrent_size = 32
dense_size     = 5
# dropout        = 0.5

model = Sequential()
model.add(Embedding(num_features, rescurrent_size))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.compile(loss = 'triplet_cosine', optimizer = 'adam')

# <<
import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def modsel():
    n_samp = train.shape[0] / 3
    sel    = 3 * np.random.choice(range(n_samp), n_samp)
    sel    = np.vstack([0 + sel, 1 + sel, 2 + sel]).T
    return np.reshape(sel, (n_samp * 3))

ms = modsel()

history = LossHistory()
fitting = model.fit(
    trn['x'][0][ms], trn['x'][0][ms], 
    nb_epoch   = 100,
    batch_size = 3 * 100,
    shuffle    = False,
    callbacks = [history]
)

preds = model.predict(awl['x'][0], verbose = True)
preds = preds / np.sqrt((preds ** 2).sum(1)[:,np.newaxis])

colors = awl['y'].argmax(1)
plt.scatter(preds[:,1], preds[:,3], c = colors)
plt.show()

# --

from sklearn.cluster import DBSCAN
db = DBSCAN(eps = .1, min_samples=400).fit(preds)

res         = sub.hash.groupby(db.labels_).apply(lambda x: x.value_counts()).reset_index()
res.columns = ('cluster', 'hash', 'cnt')
res         = res.sort('hash')

good_res = res[(res.cnt > 200) & (res.cluster > -1)]
good_res

res.hash.unique().shape[0]
good_res.hash.unique().shape[0]

eqv = list(good_res.groupby('cluster').hash.apply(lambda x: list(x)))
print_eqv(eqv, df)
