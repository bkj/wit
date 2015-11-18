import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

num_features = 1000
max_len      = 250

# --
# Helpers

def string_explode(x):
    return ' '.join(list(str(x))).strip()


def make_dataset(train_df, input_var = 'obj', response_var = 'hash'):
    global max_len
    global num_features
    
    raw_x = train_df[input_var]
    
    levs  = train_df[response_var].unique()
    raw_y = train_df[response_var]
    return {
        'x' : sequence.pad_sequences([one_hot(string_explode(x), num_features, filters = '') for x in raw_x], max_len),
        'y' : np_utils.to_categorical([list(levs).index(x) for x in raw_y]),
    }

# --
# Load data

store  = pd.HDFStore('gun_leaves.h5', complevel=9, complib='bzip2')
source = store.keys()[0]
df     = store[source]

# -- 
# Gathering schemas (could be reworked, I'm sure)

schemas = pd.DataFrame(df.groupby('id')['hash'].apply(lambda x: ' '.join(sorted(x))))
schemas['schema'] = schemas['hash']
schemas['id'] = pd.Series(schemas.index).apply(int)
del schemas['hash']

df = df.join(schemas, on = 'id', lsuffix = 'l', rsuffix = 'r')
df['id'] = df['idr']
del df['idr']
del df['idl']

# ++
# Classifier approadh -- build a classifier on one set, use it to classify the cells
# in another dataset. Problems occur when a new type of column appears.  Clustering is
# really a better approach to this problem, and a clustering approach is put
# forward in keras_pseudosiamese.py

def dset_from_schema(df, split_prop = 0.5):
    levs  = df['hash'].unique()
    dset  = make_dataset(df, input_var = 'obj', response_var = 'hash')
    sel   = np.random.uniform(0, 1, dset['x'].shape[0]) < split_prop
    
    train = { 'x' : dset['x'][sel],  'y' : dset['y'][sel] }
    val   = { 'x' : dset['x'][~sel], 'y' : dset['y'][~sel] }
    
    return {"train" : train, "val" : val, "levs" : levs}

train_schema = df['schema'].value_counts().index[0]
train_df     = df[df['schema'] == train_schema]
dset         = dset_from_schema(train_df, split_prop = 0.5)

# --
# Make model
model = Sequential()
model.add(Embedding(num_features, 128))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(levs)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")


_ = model.fit(
    dset['train']['x'], dset['train']['y'],
    batch_size      = 32,
    nb_epoch        = 4,
    validation_data = (dset['val']['x'], dset['val']['y']),
    show_accuracy   = True
)

vpreds = model.predict(dset['val']['x'])
np.mean(vpreds.argmax(1) == dset['val']['y'].argmax(1))

# --
# Predict on another schema

test_schema = df['schema'].value_counts().index[2]
test_df     = df[df['schema'] == test_schema]
tset        = dset_from_schema(test_df, split_prop = 1)

preds = model.predict(tset['train']['x'])

pd.crosstab(tset['levs'][tset['train']['y'].argmax(1)], dset['levs'][preds.argmax(1))



