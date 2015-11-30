# --
# Fake dataset generator

import re
import numpy as np
import pandas as pd
from faker import Factory

CHOICES = [
    'last_name',
    'first_name',
    'user_name',
    'street_address',
    'street_name',
    'credit_card_number',
    'address', 
    'date', 
    'iso8601',
    'ipv4', 
    'ipv6', 
    'free_email', 
    'sha256',
    'url',
    'year',
    'zipcode',
    'language_code',
    'job',
    'file_name',
    'mac_address',
    'ssn',
    'safari',
    'firefox',
    'phone_number'
]

class FakeData:
    def __init__(self, choices = CHOICES):
        self.fake    = Factory.create()
        self.choices = choices
        
    def datapoint(self):
        lab = np.random.choice(self.choices)
        val = getattr(self.fake, lab)()
        return {"hash" : lab, "obj" : val}
    
    def dataframe(self, size = 1000):
        return pd.DataFrame([self.datapoint() for i in xrange(size)])


class PairwiseData:
    
    strat = []
    keys  = {
        'hash' : 'hash', 
        'id'   : 'id',
        'obj'  : 'obj'
    }
    
    def __init__(self, df, keys = None):
        if keys:
            self.keys = keys
        
        self.df    = df
    
    def make_random(self, size = 1000):
        self.random = self.random_pairs(self.df, size)
        return self.random
    
    def make_strat(self, n_pos = 250, neg_prop = 1):
        self.pos   = self.make_pos(self.df, n_pos)
        self.neg   = self.make_neg(self.df, neg_prop)
        self.strat = pd.concat([self.pos, self.neg])
        return self.strat
        
    def make_pos(self, df, n_pos):
        print '-- making pos -- '
        tmp = df.groupby(self.keys['hash']).apply(lambda x: self.random_pairs(x, n_pos))
        
        tmp = tmp[tmp.id1 != tmp.id2]
        tmp = tmp.drop_duplicates().reset_index()
        
        del tmp[self.keys['hash']]
        del tmp['level_1']
        
        return tmp
    
    # NB : This makes all negative pairs.  Might be better to sample here.
    def make_neg(self, df, neg_prop = 1):
        print '-- making neg --'
        ids = list(df[self.keys['id']].sample( np.floor(neg_prop * df.shape[0]), replace = False))
        tmp = df[df.id.apply(lambda x: x in ids)]
        tmp = tmp.groupby(self.keys['id']).apply(self.all_neg_pairs)
        tmp = tmp.drop_duplicates().reset_index()
        
        # Don't push the same string apart? I don't know what to do about this.
        # I think the impact of this is an artifact of the "pseduosiamese"
        # architecture I'm using, and will be fixed when I incorporate the
        # actual siamese architecture.
        # tmp = tmp[tmp.obj1 != tmp2.obj2]
        # tmp = tmp.reset_index()
        
        del tmp[self.keys['id']]
        del tmp['level_1']
        return tmp
    
    def random_pairs(self, x, size):
        s1 = x.sample(size, replace = True).reset_index()
        s2 = x.sample(size, replace = True).reset_index()
        return pd.DataFrame(data = {
            "id1"    : s1[self.keys['id']],
            "id2"    : s2[self.keys['id']],
            
            "obj1"   : s1[self.keys['obj']],
            "obj2"   : s2[self.keys['obj']],
            
            "hash1"  : s1[self.keys['hash']],
            "hash2"  : s2[self.keys['hash']],
            
            "match"  : (s1[self.keys['hash']] == s2[self.keys['hash']]) + 0
        })
        
    def all_neg_pairs(self, x):
        out = []
        tmp = x.apply(
            lambda s1: x.apply(
                lambda s2: out.append({
                    "id1"   : s1[self.keys['id']],
                    "id2"   : s2[self.keys['id']],
                    "hash1" : s1[self.keys['hash']],
                    "hash2" : s2[self.keys['hash']],
                    "obj1"  : s1[self.keys['obj']],
                    "obj2"  : s2[self.keys['obj']],
                    "match" : 0,
                }), 1
            ), 1
        )
        
        out = pd.DataFrame(out)
        
        # Remove information pushing certain hashes apart
        out = out[out.hash1 != out.hash2] 
        return out

# --
# Formatting / featurizing for Keras input

from time import time

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot

class KerasFormatter:
    
    levs = None
    
    def __init__(self, num_features = 1000, max_len = 150, levs = None):
        self.num_features = num_features
        self.max_len      = max_len
        if levs:
            self.levs = levs
    
    def format(self, data, xfields, yfield, words = False):
        if not isinstance(xfields, list):
            raise Exception('xfields must be a list')
        
        if len(xfields) > 2:
            raise Exception('illegal number of xfields')
        
        xs = [self._format_x(data[xf], words) for xf in xfields]
        
        if not self.levs:
            self.levs = sorted(list(data[yfield].unique()))
        
        y = self._format_y(data[yfield], self.levs)
        return ({'x' : xs, 'y' : y}, self.levs)
    
    # def format_with_val(self, data, xfields, yfield, val_prop = 0.2):
    #     sel       = np.random.uniform(0, 1, data.shape[0]) > val_prop
    #     self.levs = sorted(list(data[yfield].unique()))
    #     return (
    #         self.format(data[sel], xfields, yfield), 
    #         self.format(data[~sel], xfields, yfield), 
    #         self.levs
    #     )
    
    # def format_symmetric_with_val(self, data, xfields, yfield, val_prop = 0.2):
    #     sel       = np.random.uniform(0, 1, data.shape[0]) > val_prop
    #     self.levs = sorted(list(data[yfield].unique()))
    #     return (
    #         self.format_symmetric(data[sel], xfields, yfield), 
    #         self.format_symmetric(data[~sel], xfields, yfield), 
    #         self.levs
    #     )
    
    def format_symmetric(self, data, xfields, yfield, words = False):
        self.levs = None
        tmp = self.format(data, xfields, yfield, words)
        
        tx0 = np.concatenate([tmp['x'][0], tmp['x'][1]])
        tx1 = np.concatenate([tmp['x'][1], tmp['x'][0]])
        
        tmp['x'][0] = tx0
        tmp['x'][1] = tx1
        
        tmp['y'] = np.concatenate([tmp['y'], tmp['y']])
        return (tmp, self.levs)
    
    def _format_x(self, z, words):
        return sequence.pad_sequences(
            [one_hot(string_explode(x, words = words), self.num_features, filters = '') for x in z], 
        self.max_len)
    
    def _format_y(self, z, levs):
        return np_utils.to_categorical([levs.index(x) for x in z])


# --
# Neural network string classifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM, JZS1
from keras.layers.embeddings import Embedding

class WitClassifier:
    model = None
    
    def __init__(self, train, levs, model = None):        
        self.train = train    
        self.levs  = levs
        
        self.intuit_params()
        
        if model:
            self.model = model
        else:
            self.model = self.compile()
    
    def intuit_params(self):            
        self.n_classes    = len(self.levs)
        self.num_features = self.train['x'][0].max() + 1
        self.max_len      = self.train['x'][0].shape[1]
    
    def predict(self, data, verbose = 1):
        return self.model.predict(data, verbose = verbose)

# --
# String classifier
class StringClassifier(WitClassifier):
    
    recurrent_size = 128
    dropout        = 0.5
    
    def compile(self):
        model = Sequential()
        model.add(Embedding(self.num_features, self.recurrent_size))
        model.add(LSTM(self.recurrent_size))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")
        return model
    
    def fit(self, batch_size = 100, nb_epoch = 10):
        if len(self.train['x']) > 1:
            raise Exception('train[x] has more than one entry -- stopping')
        
        _ = self.model.fit(
            self.train['x'][0], self.train['y'],
            batch_size       = batch_size,
            nb_epoch         = nb_epoch,
            validation_split = 0.2,
            show_accuracy    = True
        )
        
        return True
    
    def classify_string(self, string):
        num_features = 1000
        max_len      = 50
        z            = np.array(one_hot(string_explode(string), num_features, filters = ''))
        z.shape      = (1, z.shape[0])
        z            = sequence.pad_sequences(z, max_len)
        return self.levs[self.model.predict(z, batch_size = 1).argmax()]

# --
# Siamese network trainer
class SiameseClassifier(WitClassifier):
    
    recurrent_size = 64
    dense_size     = 32
    dropout        = 0.25
    
    # -- Define Model
    def _make_leg(self):
        leg = Sequential()
        leg.add(Embedding(self.num_features, self.recurrent_size))
        # Two layer
        # leg.add(JZS1(self.recurrent_size, return_sequences = True))
        leg.add(JZS1(self.recurrent_size))
        leg.add(Dense(self.dense_size))
        return leg
    
    def compile(self):
        print '--- compiling siamese model ---'
        model = Sequential()
        model.add(Merge([self._make_leg(), self._make_leg()], mode='dot'))
        model.add(Dropout(self.dropout))
        model.add(Dense(2)) # Hardcoded for now
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    
    def fit(self, batch_size = 100, nb_epoch = 10):
        T = time()
        _ = self.model.fit(
            self.train['x'], self.train['y'],
            batch_size       = batch_size,
            nb_epoch         = nb_epoch,
            validation_split = 0.2,
            show_accuracy    = True
        )
        
        print 'elapsed time :: %f' % (time() - T) 
        return True

# --
# Helpers

def string_explode(x, words = False):
    if not words:
        return ' '.join(list(str(x))).strip()
    elif words:
        tmp = re.sub('([^\w])', ' \\1 ', x)
        tmp = re.sub(' +', ' ', tmp)
        return tmp

# Helper function for making testing data
def strat_pairs(df, n_match = 100, n_nonmatch = 10, hash_id = 'hash'):
    print 'strat_pairs -- starting'
    
    out = []
    uh  = df[hash_id].unique()
    ds  = dict([(u, df[df[hash_id] == u]) for u in uh])
    for u1 in uh:
        d1 = ds[u1]
        print 'strat_pairs :: %s' % u1
        
        for u2 in uh:
            d2  = ds[u2]
            
            cnt = n_match if (u1 == u2) else n_nonmatch
            s1  = d1.sample(cnt, replace = True).reset_index()
            s2  = d2.sample(cnt, replace = True).reset_index()
            
            # If we were using this for training, 
            # we'd want to remove these because they're non-informative
            # not_same = s1.obj != s2.obj
            # s1       = s1[not_same]
            # s2       = s2[not_same]
            
            out.append(pd.DataFrame(data = {
                "obj1"   : s1['obj'],
                "obj2"   : s2['obj'],
                
                "hash1"  : s1[hash_id],
                "hash2"  : s2[hash_id],
                
                "match"  : (s1[hash_id] == s2[hash_id]) + 0
            }))
    
    return pd.concat(out)


# Helper function for viewing aggregate similarity between fields
def make_self_sims(x):
    tmp = x.groupby(['hash1', 'hash2'])['preds']
    sims = pd.DataFrame({
        'sim' : tmp.agg(np.median),
        'cnt' : tmp.agg(len),
        'sum' : tmp.agg(sum)
    }).reset_index()
    
    sims.sim  = sims.sim.round(4)
    self_sims = sims[sims['hash1'] == sims['hash2']].sort('sim')
    return self_sims, sims


def make_equiv(test, THRESH = .8):
    equivs  = {}
    uequivs = {}
    uhash   = test['hash1'].unique()
    for h in uhash:
        sub1      = test[test['hash1'] == h]
        medpred1  = sub1.groupby('hash2')['preds'].agg(np.median)
        
        sub2      = test[test['hash2'] == h]
        medpred2  = sub2.groupby('hash1')['preds'].agg(np.median)
        
        e = sorted(list(set(medpred1.index[medpred1 > THRESH]).union(set(medpred2.index[medpred2 > THRESH]))))
        print '%s :: %d' % (h, len(e))
        uequivs[str(e)] = e
        equivs[h]       = e
    
    return equivs, uequivs


class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'


def print_eqv(eqv, df):
    for e in eqv:
        print bcolors.WARNING + '\n --- \n'
        print e
        print '\n' + bcolors.ENDC
        for h in e:
            print bcolors.OKGREEN + h + '\t(%d rows)' % df[df.hash == h].shape[0] + bcolors.ENDC
            print df[df.hash == h].obj.head()
