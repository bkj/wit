# --
# Fake dataset generator

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
    
    strat = None
    keys  = {
        'hash' : 'hash', 
        'id'   : 'id',
        'obj'  : 'obj'
    }
    
    def __init__(self, df, nsamp = 250):
        self.nsamp = nsamp
        self.df    = df
    
    def make_random(self):
        self.random = self.random_pairs(self.df)
        return self.random
        
    def make_strat(self):
        self.pos   = self.make_pos(self.df)
        self.neg   = self.make_neg(self.df)
        self.strat = pd.concat([self.pos, self.neg])
        return self.strat
    
    def make_dstrat(self, neg = True, pos = False, prop = .1):
        if not self.strat:
            _ = self.make_strat()
        
        tmpneg = self.neg.sample(np.floor(prop * self.neg.shape[0])) if neg else self.neg
        tmppos = self.pos.sample(np.floor(prop * self.pos.shape[0])) if pos else self.pos
        
        return pd.concat([tmppos, tmpneg])
        
    def make_pos(self, df):
        print '-- making pos -- '
        tmp = df.groupby(self.keys['hash']).apply(self.random_pairs)
        tmp = tmp.drop_duplicates().reset_index()
        del tmp[self.keys['hash']]
        del tmp['level_1']
        return tmp
        
    def make_neg(self, df):
        print '-- making neg --'
        tmp = df.groupby(self.keys['id']).apply(self.all_neg_pairs)
        tmp = tmp.drop_duplicates().reset_index()
        del tmp[self.keys['id']]
        del tmp['level_1']
        return tmp
    
    def random_pairs(self, x):
        s1 = x.sample(self.nsamp, replace = True).reset_index()
        s2 = x.sample(self.nsamp, replace = True).reset_index()
        return pd.DataFrame(data = {
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
                    "hash1" : s1[self.keys['hash']],
                    "hash2" : s2[self.keys['hash']],
                    "obj1"  : s1[self.keys['obj']],
                    "obj2"  : s2[self.keys['obj']],
                    "match" : 0,
                }), 1
            ), 1
        )
        
        out = pd.DataFrame(out)
        return out[out.hash1 != out.hash2]

# --
# Formatting / featurizing for Keras input

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot

class KerasFormatter:
    
    levs = None
    
    def __init__(self, num_features = 1000, max_len = 50, levs = None):
        self.num_features = num_features
        self.max_len      = max_len
        if levs:
            self.levs = levs
    
    def format(self, data, xfields, yfield, words = False):
        if not isinstance(xfields, list):
            raise Exception('xfields must be a list')
        
        if len(xfields) > 2:
            raise Exception('illegal number of xfields')
        
        xs   = [self._format_x(data[xf], words) for xf in xfields]
        
        if not self.levs:
            self.levs = sorted(list(data[yfield].unique()))
        
        y    = self._format_y(data[yfield], self.levs)
        return {'x' : xs, 'y' : y}
    
    def format_with_val(self, data, xfields, yfield, val_prop = 0.2):
        sel       = np.random.uniform(0, 1, data.shape[0]) > val_prop
        self.levs = sorted(list(data[yfield].unique()))
        return (
            self.format(data[sel], xfields, yfield), 
            self.format(data[~sel], xfields, yfield), 
            self.levs
        )
    
    def format_symmetric_with_val(self, data, xfields, yfield, val_prop = 0.2):
        sel       = np.random.uniform(0, 1, data.shape[0]) > val_prop
        self.levs = sorted(list(data[yfield].unique()))
        return (
            self.format_symmetric(data[sel], xfields, yfield), 
            self.format_symmetric(data[~sel], xfields, yfield), 
            self.levs
        )
    
    def format_symmetric(self, data, xfields, yfield):
        tmp         = self.format(data, xfields, yfield)
        tmp['x'][0] = np.concatenate([tmp['x'][0], tmp['x'][1]])
        tmp['x'][1] = np.concatenate([tmp['x'][1], tmp['x'][0]])        
        tmp['y']    = np.concatenate([tmp['y'], tmp['y']])
        return tmp
    
    def _format_x(self, z, words):
        return sequence.pad_sequences(
            [one_hot(self.string_explode(x, words = words), self.num_features, filters = '') for x in z], 
        self.max_len)
    
    def _format_y(self, z, levs):
        return np_utils.to_categorical([levs.index(x) for x in z])
    
    def string_explode(self, x, words):
        if not words:
            return ' '.join(list(str(x))).strip()
        elif words:
            tmp = re.sub('([^\w])', ' \\1 ', x)
            tmp = re.sub(' +', ' ', tmp)
            return tmp





# --
# Neural network string classifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

class WitClassifier:
    model = None
    
    def __init__(self, train, val, levs, model = None):        
        self.train = train
        self.val   = val
        self.levs  = levs
        
        self.intuit_params()
        
        if model:
            self.model = model
        else:
            self.model = self.compile()
    
    def intuit_params(self):            
        self.n_classes    = len(levs)
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
        print '--- compiling string classifier model ---'
        model = Sequential()
        model.add(Embedding(self.num_features, self.recurrent_size))
        model.add(LSTM(self.recurrent_size))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")
        return model
    
    def fit(self, batch_size = 100, nb_epoch = 10):
        if len(self.train['x'] > 1):
            raise Exception('train[x] has more than one entry -- stopping')
        
        _ = self.model.fit(
            self.train['x'][0], self.train['y'],
            batch_size      = batch_size,
            nb_epoch        = nb_epoch,
            validation_data = (self.val['x'][0], self.val['y']),
            show_accuracy   = True
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
    dropout        = 0.5
    
    # -- Define Model
    def _make_leg(self):
        leg = Sequential()
        leg.add(Embedding(num_features, self.recurrent_size))
        leg.add(LSTM(self.recurrent_size))
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
        _ = self.model.fit(
            self.train['x'], self.train['y'],
            batch_size      = batch_size,
            nb_epoch        = nb_epoch,
            validation_data = (self.val['x'], self.val['y']),
            show_accuracy   = True
        )
        
        return True

