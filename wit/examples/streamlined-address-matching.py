# --
# Load deps

import keras
import pandas as pd
from fuzzywuzzy import fuzz
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
formatter    = KerasFormatter(num_features, max_len)

# --
# Load and format data

df       = pd.read_csv('/Volumes/phronesis/address/real_address.csv')
df['id'] = 0

# --
# Make all pairs

df.hash.unique().shape # number of hashes

train  = make_triplet_train(df.head(60000), N = 20)
train.to_csv('/Volumes/phronesis/address/train_address.csv')

# -- EDIT : Removing duplicates --
tmp   = train.groupby('ex').apply(lambda x: x.obj[x.role == 'anc'] == x.obj[x.role == 'pos']).reset_index()
train = train[train.ex.isin(list(tmp.level_1[tmp.obj]))]

# --

trn, levs  = formatter.format(train, ['obj'], 'hash')

classifier = TripletClassifier(trn, levs)
classifier.fit(batch_size = 250, nb_epoch = 3)



awl_sub = df.tail(5000)
awl, _  = formatter.format(awl_sub, ['obj'], 'hash')
preds = model.predict(awl['x'][0], verbose = True)

out   = {}
uhash = awl_sub.hash.unique()
for i in range(len(uhash)):
    tmp    = preds[np.array(awl_sub.hash == uhash[i])]
    out[i] = np.dot(tmp, tmp.T)

sims = map(lambda x: np.mean(x), out.values())

def get_sim(a, b):
    preds = model.predict(formatter._format_x([a, b], False))
    return np.dot(preds, preds.T)[0, 1]

def compare(a, b):
    learned_sim = get_sim(a, b)
    fuzz_sim    = fuzz.ratio(a, b)
    print '\nlearned_sim : %f \t| fuzz_sim : %f\n' %(learned_sim, fuzz_sim)
    return learned_sim, fuzz_sim

_ = compare('101 fake street', '101 fake st')
_ = compare('101 fake street', '102 fake street')
_ = compare('101 fake street', '102 fake st')




# --
# Comparison to Levenshtein

from fuzzywuzzy import fuzz

out  = []
prev = None
awl_sub = awl_sub.reset_index()
for i, r in awl_sub.iterrows():
    print i
    tmp  = dict(r)
    
    if prev:
        out.append((
            tmp['hash'] == prev['hash'],
            fuzz.ratio(tmp['obj'], prev['obj']),
            np.dot(
                preds[i], preds[i - 1]
            )
        ))
    
    prev = tmp

res = pd.DataFrame(out)
res.columns = ('same', 'fuzz', 'wit')
res.wit = res.wit * 100

plt.hist(np.array(res.fuzz[~res.same]), 100)
plt.hist(np.array(res.fuzz[res.same]), 100)
plt.show()


# --
from sklearn import metrics

# --
# Learned
y    = np.array(res.same) + 0
pred = np.array(res.wit)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
metrics.auc(fpr, tpr)

# --
# Fuzz
y    = np.array(res.same) + 0
pred = np.array(res.fuzz)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
metrics.auc(fpr, tpr)


s1 = '19 stanton court'
s2 = '18 stanton court'

fuzz.ratio(s1, s2)
get_sim(s1, s2)

