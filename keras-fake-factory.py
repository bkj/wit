import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

from pprint import pprint
from faker import Factory

fake = Factory.create()
fake.phone_number()

choices = [
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

num_features = 1000

# --
# Helpers

def string_explode(x):
    return ' '.join(list(str(x))).strip()


def fake_datapoint(choices):
    lab = np.random.choice(choices)
    val = getattr(fake, lab)()
    return (lab, val)


def make_dataset(choices, SIZE = 1000):
    orig = [fake_datapoint(choices) for i in range(SIZE)]
    return {
        'y'    : np_utils.to_categorical([choices.index(x[0]) for x in orig]),
        'x'    : sequence.pad_sequences([one_hot(string_explode(x[1]), num_features, filters = '') for x in orig]),
        'orig' : orig
    }


def classify_string(string):
    z = one_hot(string_explode(string), num_features, filters = '')
    z = np.array(z)
    z.shape = (1, z.shape[0])
    z = sequence.pad_sequences(z, 128)
    return choices[model.predict(z, batch_size = 1).argmax()]


# --

print('building model...')
model = Sequential()
model.add(Embedding(num_features, 128))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(choices)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")


print('making data...')
train = make_dataset(choices, 50000)
val   = make_dataset(choices, 1000)


print('training...')
_ = model.fit(
    train['x'], train['y'],
    batch_size      = 100,
    nb_epoch        = 4,
    validation_data = (val['x'], val['y']),
    show_accuracy   = True
)


test  = make_dataset(choices, 1000)
preds = model.predict(test['x'])
pd.crosstab(np.array(choices)[preds.argmax(1)], np.array(choices)[test['y'].argmax(1)])


pd.Series(map(len, [fake.phone_number() for i in range(5000)])).value_counts()
pd.Series(map(len, [fake.ssn() for i in range(5000)])).value_counts()


# # --
# Tests

classify_string('ben@gophronesis.com')
classify_string('12.3.45.678')

classify_string('2000-01-01')
classify_string('123-45-6789')
classify_string('123-456-7890')

classify_string('http://www.google.com')


