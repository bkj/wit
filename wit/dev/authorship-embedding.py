import pandas as pd
import urllib2

from pprint import pprint
from matplotlib import pyplot as plt

from bs4 import BeautifulSoup
from hashlib import md5

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

np.set_printoptions(linewidth=250)

# May need to add things here to make this run the same way each time
np.random.seed(123)

# --

num_features = 10000 # Words
max_len      = 100   # Words

formatter = KerasFormatter(num_features, max_len)

# --
# Load data
orig         = pd.read_csv('/Users/BenJohnson/projects/laundering/sec/edward/analysis/crowdsar/crowdsar_user.csv', sep = '|', header = None)
orig.columns = ('hash', 'obj')
orig['id']   = 0

# Get 
frequent_posters = orig.hash.value_counts().head(100).index
nfrequent_posters = orig.hash.value_counts().head(100).tail(25).index

sub = orig[orig.hash.isin(frequent_posters)]
sel = np.random.uniform(0, 1, sub.shape[0]) > .9
sub = sub[sel].drop_duplicates()

sel2 = np.random.uniform(0, 1, sub.shape[0]) > .5
df   = sub[sel2]
tdf  = sub[~sel2]

tdf2 = orig[orig.hash.isin(nfrequent_posters)].drop_duplicates()
sel3 = np.random.uniform(0, 1, tdf2.shape[0]) > .9
tdf2 = tdf2[sel3]



# --

train = make_triplet_train(df, N = 500)

trn, trn_levs = formatter.format(train, ['obj'], 'hash')
awl, awl_levs = formatter.format(train.drop_duplicates(), ['obj'], 'hash')

# tst, tst_levs = formatter.format(tdf, ['obj'], 'hash')
out, out_levs = formatter.format(tdf2, ['obj'], 'hash')

# --
# Define model

recurrent_size = 64
dense_size     = 16

model = Sequential()
model.add(Embedding(num_features, recurrent_size))
model.add(LSTM(recurrent_size, return_sequences = True))
model.add(LSTM(recurrent_size))
model.add(Dense(dense_size))
model.add(Activation('unit_norm'))
model.compile(loss = 'triplet_cosine', optimizer = 'adam')

# --
# Train model

for i in range(60):
    ms = modsel(train.shape[0], N = 3)
    fitting = model.fit(
        trn['x'][0][ms], trn['x'][0][ms], 
        nb_epoch   = 3,
        batch_size = 3 * 250,
        shuffle    = False
    )

json_string = model.to_json()
open('author2_architecture.json', 'w').write(json_string)
model.save_weights('author2_weights.h5')

tr_preds = model.predict(awl['x'][0], verbose = True, batch_size = 250)

colors = awl['y'].argmax(1)
plt.scatter(tr_preds[:,0], tr_preds[:,1], c = colors)
plt.show()


# ------------------------------------------------
# Load pretrained model
#
# from keras.models import model_from_json
# model = model_from_json(open('author_architecture.json').read())
# model.load_weights('author_weights.h5')

# <<

shp  = awl['y'].shape[1]
amax = awl['y'].argmax(1)
sims = np.zeros( (awl['y'].shape[1], awl['y'].shape[1]) )
tmps = [tr_preds[amax == i] for i in range(shp)]

for i in range(shp):
    print i
    a = tmps[i]
    for j in range(shp):
        b  = tmps[j]
        mn = np.mean(np.dot(a, b.T) > .8)
        sims[i][j] = mn


np.mean(np.max(sims, 0) - np.diag(sims))
np.mean(np.max(sims, 0) - sims)

np.mean(sims.argmax(1) == np.arange(sims.shape[0]))



# >>

ts_preds = model.predict(tst['x'][0], verbose = True, batch_size = 250)

tmpsel = np.random.choice(ts_preds.shape[0], 5000)
sim    = np.dot(ts_preds[tmpsel], tr_preds.T)

np.mean(tst['y'].argmax(1)[tmpsel] == awl['y'].argmax(1)[sim.argmax(1)])

tdf[]

# --

out_preds = model.predict(out['x'][0], verbose = True, batch_size = 250)

outsims = np.dot(out_preds, out_preds.T)

shp  = out['y'].shape[1]
amax = out['y'].argmax(1)
sims = np.zeros( (out['y'].shape[1], out['y'].shape[1]) )
tmps = [out_preds[amax == i] for i in range(shp)]

for i in range(shp):
    print i
    a = tmps[i]
    for j in range(shp):
        b  = tmps[j]
        mn = np.mean(np.dot(a, b.T) > .8)
        sims[i][j] = mn

sims.argmax(1) == np.arange(sims.shape[0])

np.fill_diagonal(outsims, 0)
rowmax = outsims.argmax(1)
by_user = map(lambda K: np.mean(amax[rowmax[amax == K]] == K), range(out['y'].shape[1]))

pprint(by_user)

# >>

from sklearn.cluster import KMeans

lens = np.array(tdf2.obj.apply(lambda x: len(str(x))))

km = KMeans(n_clusters = 26)
cl = km.fit_predict(out_preds[lens > 100])

amax = out['y'][lens > 100].argmax(1)
pd.crosstab(cl, amax)





# <<

# --

out_preds = model.predict(out['x'][0], verbose = True, batch_size = 250)

sel     = np.random.uniform(0, 1, out_preds.shape[0]) > .5
outsims = np.dot(out_preds[sel], out_preds[~sel].T)

amax1 = out['y'].argmax(1)[sel]
amax2 = out['y'].argmax(1)[~sel]

conf = pd.crosstab(amax1, amax2[outsims.argmax(1)])
np.mean(np.array(conf).argmax(1) == range(conf.shape[0]))
