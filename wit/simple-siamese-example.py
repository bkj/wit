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

df           = pd.melt(raw_df, id_vars = 'id')
df.columns   = ('id', 'hash', 'obj')

train  = make_triplet_train(df)
trn, _ = formatter.format(train, ['obj'], 'hash')
awl, _ = formatter.format(df, ['obj'], 'hash')

# --
# Compile and train classifier
recurrent_size = 32
dense_size     = 4

model = Sequential()
model.add(Embedding(num_features, recurrent_size))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.compile(loss = 'triplet_cosine', optimizer = 'adam')

ms = modsel(train.shape[0], N = 3)
_ = model.fit(
    trn['x'][0][ms], trn['x'][0][ms], 
    nb_epoch   = 10,
    batch_size = 3 * 250,
    shuffle    = False
)

preds = model.predict(awl['x'][0], verbose = True)
preds = preds / np.sqrt((preds ** 2).sum(1)[:,np.newaxis])

colors = awl['y'].argmax(1)
plt.scatter(preds[:,0], preds[:,1], c = colors)
plt.show()

# --
# Cluster 

from sklearn.cluster import DBSCAN
db = DBSCAN(eps = .1).fit(preds)

res         = df.hash.groupby(db.labels_).apply(lambda x: x.value_counts()).reset_index()
res.columns = ('cluster', 'hash', 'cnt')
res         = res.sort('hash')

good_res = res[(res.cnt > 100) & (res.cluster > -1)]
good_res

sorted(res.hash.unique())
sorted(good_res.hash.unique())

eqv = list(good_res.groupby('cluster').hash.apply(lambda x: list(x)))
print_eqv(eqv, df)






