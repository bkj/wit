# --
# Load deps

import keras
import pandas as pd
import urllib2

from hashlib import md5
from bs4 import BeautifulSoup
from pprint import pprint
from matplotlib import pyplot as plt

import sys
sys.path.append('/Users/BenJohnson/projects/what-is-this/wit/')
from wit import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

np.set_printoptions(linewidth=100)

# -- 
# Config + Init

num_features = 75  # Character
max_len      = 350 # Character

formatter = KerasFormatter(num_features, max_len)

# --
# Load and format data

path     = '/Users/BenJohnson/projects/what-is-this/qpr/gun_leaves_20151118_v2.h5'
in_store = pd.HDFStore(path,complevel=9, complib='bzip2')

df = {}
for k in ['/www.remingtonsociety.com', '/marauderairrifle.com']:
    df[k]           = in_store[k]
    df[k]['origin'] = k

df = pd.concat(df.values())
in_store.close()

# Post cleaning
df['shash'] = df.origin.apply(lambda x: md5(x).hexdigest()[0:5])
df['hash']  = df.apply(lambda x: str(x['hash']) + '-' + x['shash'], 1)
df['id']    = df.apply(lambda x: str(x['id']) + '-' + x['shash'], 1)
df['src']   = df.obj
df['obj']   = df.src.apply(lambda x: BeautifulSoup(x).text.encode('ascii', 'ignore'))

# Subset to frequent paths, w/ more than 100 unique values
chash = df.groupby('hash').apply(lambda x: len(x.obj.unique()))
keep  = list(chash[chash > 100].index)
df    = df[df.hash.apply(lambda x: x in keep)]


# --
# Make training data
train     = make_triplet_train(df, N = 600)
trn, levs = formatter.format(train, ['obj'], 'hash')

classifier = TripletClassifier(trn, levs)
classifier.fit(batch_size = 250, nb_epoch = 3)

# --
# Test dataset
unq = df.copy()
del unq['id']
unq    = unq.drop_duplicates()
awl, _ = formatter.format(unq, ['obj'], 'hash')
preds  = classifier.predict(awl['x'][0], verbose = True)
