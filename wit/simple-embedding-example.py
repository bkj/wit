import urllib2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from wit import *

# --

num_features = 100 # Characters
max_len      = 150 # Characters

formatter    = KerasFormatter(num_features, max_len)

# -- 
# Training

# Load data
url          = 'https://raw.githubusercontent.com/chrisalbon/variable_type_identification_test_datasets/master/datasets_raw/ak_bill_actions.csv'

raw_df       = pd.read_csv(url)
sel          = np.random.choice(range(raw_df.shape[0]), 1000)
raw_df       = raw_df.iloc[sel].reset_index()
raw_df['id'] = range(raw_df.shape[0])

# Subset to a couple of types
raw_df = raw_df[['id', 'bill_id', 'date', 'action']]

df           = pd.melt(raw_df, id_vars = 'id')
df.columns   = ('id', 'hash', 'obj')

train  = make_triplet_train(df, N = 500)
trn, _ = formatter.format(train, ['obj'], 'hash')
awl, _ = formatter.format(df, ['obj'], 'hash')

# --
# Compile and train classifier
recurrent_size = 32
dense_size     = 2

model = Sequential()
model.add(Embedding(num_features, recurrent_size))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.add(Activation('unit_norm'))
model.compile(loss = 'triplet_cosine')

ms = modsel(train.shape[0], N = 3)
_ = model.fit(
    trn['x'][0][ms], trn['x'][0][ms], 
    nb_epoch   = 5,
    batch_size = 3 * 250,
    shuffle    = False
)

preds = model.predict(awl['x'][0], verbose = True)

colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,1], c = colors)
plt.show()