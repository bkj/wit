# --

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot

class KerasFormatter():
    
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
        sel  = np.random.uniform(0, 1, data.shape[0]) > val_prop
        self.levs = sorted(list(data[yfield].unique()))
        return self.format(data[sel], xfields, yfield), self.format(data[~sel], xfields, yfield), self.levs
        
    def format_symmetric(self, data, xfields, yfield):
        tmp         = self.form(data, xfield, yfield)
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

class FakeData():
    def __init__(self, choices = CHOICES):
        self.fake    = Factory.create()
        self.choices = choices
        
    def datapoint(self):
        lab = np.random.choice(self.choices)
        val = getattr(fake, lab)()
        return {"hash" : lab, "obj" : val}
    
    def dataframe(self, size = 1000):
        return pd.DataFrame([self.datapoint() for i in xrange(size)])

# --

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

class StringClassifier():
    
    model          = None
    recurrent_size = 128
    dropout        = 0.5
    
    def __init__(self, train, val, levs, model = None):
        if len(train['x']) > 1:
            raise Exception('train[x] is malformatted (larger than 1)')
        
        train['x'] = train['x'][0]
        val['x']   = val['x'][0]
        
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
        self.num_features = self.train['x'].max() + 1
        self.max_len      = self.train['x'].shape[1]
    
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
        _ = self.model.fit(
            self.train['x'], self.train['y'],
            batch_size      = batch_size,
            nb_epoch        = nb_epoch,
            validation_data = (self.val['x'], self.val['y']),
            show_accuracy   = True
        )
        
        return True
    
    def predict(self, data, verbose = 1):
        return self.model.predict(data, verbose = verbose)

# --
# Example

num_features = 1000
max_len      = 50

f         = FakeData()
formatter = KerasFormatter(num_features, max_len)

data             = f.dataframe(size = 5000)
train, val, levs = formatter.format_with_val(data, ['obj'], 'hash')

classifier = StringClassifier(train, val, levs, model)
classifier.fit()

testdata = f.dataframe(size = 5000)
test     = formatter.format(testdata, ['obj'], 'hash')

preds = classifier.predict(test['x'])
pd.crosstab(preds.argmax(1), test['y'].argmax(1))
