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

train_data = df.groupby('shash').apply(lambda x: strat_pairs(x, n_match = 500, n_nonmatch = 125)).drop_duplicates().reset_index()


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
    batch_size      = 250,
    nb_epoch        = 100,
    validation_data = ([val['x1'], val['x2']], val['y']),
    show_accuracy   = True
)

# Eval on train
preds          = model.predict([trn['x1'], trn['x1']], verbose = 1)
preds.shape    = (preds.shape[0],)
train['preds'] = preds[:train.shape[0]]

make_self_sims(train)

# Eval on test
test = strat_pairs(df, n_nonmatch = 25, n_match = 25)
tst  = make_dataset(test)

preds         = model.predict([tst['x1'], tst['x2']], verbose = 1)
preds.shape   = (preds.shape[0], )
test['preds'] = preds[:test.shape[0]]

make_self_sims(test)
