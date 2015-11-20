import re

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from pprint import pprint
from hashlib import md5

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from bs4 import BeautifulSoup

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)


# --
# Catting all datasets

out_store  = pd.HDFStore('gun_domains.h5',complevel=9, complib='bzip2')

df = {}
for k in out_store.keys():
    tmp = out_store[k]
    tmp['source'] = k
    df[k] = tmp

df = pd.concat(df).reset_index()
del df['level_0']
del df['level_1']

df['shash']   = df.apply(lambda x: md5(str(x['source'])).hexdigest()[0:5], 1)
df['hash']    = df.apply(lambda x: md5(str(x['source'])).hexdigest()[0:5] + '-' + str(x['field_id']), 1)
df['sdoc_id'] = df.apply(lambda x: md5(str(x['source'])).hexdigest()[0:5] + '-' + str(x['doc_id']), 1)

df['markup'] = df['obj']
df['obj']    = df.markup.apply(lambda x: BeautifulSoup(x).text.encode('utf-8'))

pos = df.groupby('hash').apply(lambda x: random_pairs(x, nsamp = 250)).drop_duplicates().reset_index()
del pos['hash']
del pos['level_1']

orig_neg = df.groupby('id').apply(all_neg_pairs).drop_duplicates()
neg      = orig_neg.sample(20000).reset_index()
del neg['id']
del neg['level_1']

train_data = pd.concat([pos, neg])

# Random sampling
#train_data = df.groupby('shash').apply(lambda x: strat_pairs(x, n_match = 500, n_nonmatch = 125)).drop_duplicates().reset_index()

# --

sel   = np.random.uniform(0, 1, train_data.shape[0]) > .2

train = train_data.iloc[sel]
trn   = make_dataset(train, words = False)

valid = train_data.iloc[~sel]
val   = make_dataset(valid, words = False)

# --

model = make_model()
_ = model.fit(
    [trn['x1'], trn['x2']], trn['y'], 
    batch_size      = 100,
    nb_epoch        = 30,
    validation_data = ([val['x1'], val['x2']], val['y']),
    show_accuracy   = True
)

# Eval on train
preds          = model.predict([trn['x1'], trn['x1']], verbose = 1)
preds.shape    = (preds.shape[0],)
train['preds'] = preds[:train.shape[0]]

self_sims, sims = make_self_sims(train)
self_sims


# Eval on test
test = strat_pairs(df, n_nonmatch = 25, n_match = 25)
tst  = make_dataset(test)

preds         = model.predict([tst['x1'], tst['x2']], verbose = 1)
preds.shape   = (preds.shape[0], )
test['preds'] = preds[:test.shape[0]]

self_sims, sims = make_self_sims(test)

equivs, uequivs = make_equiv(test, THRESH = .8)
eqv             = uequivs.values()

assert(len(np.unique(np.hstack(eqv))) == len(np.hstack(eqv)))

print_eqv(eqv)