import re

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from pprint import pprint
from hashlib import md5

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

# --

def replicate(x):
    return np.vstack([x, x, x, x, x, x, x, x, x, x]).T

num_features = 3

x1 = replicate(np.random.choice(range(num_features), 5000))
x2 = replicate(np.random.choice(range(num_features), 5000))
y  = ((x1 == x2).sum(axis = 1) > 0) + 0

x1_neg = x1[y == 0]
x2_neg = x2[y == 0]
y_neg  = y[y == 0]

x1_fake = replicate(np.random.choice(range(num_features), 1000))
x2_fake = replicate(np.random.choice(range(num_features), 1000))
y_fake  = np.ones(x1_fake.shape[0])

x1_train = np.vstack([x1_neg, x1_fake])
x2_train = np.vstack([x2_neg, x2_fake])
y_train  = np.hstack([y_neg, y_fake])

# -- Define Model
def make_leg():
    leg = Sequential()
    leg.add(Embedding(num_features, 10))
    leg.add(LSTM(10))
    leg.add(Dense(5))
    return leg

model = Sequential()

model.add(Merge([make_leg(), make_leg()], mode='dot'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# -- Train Model
_ = model.fit(
    [x1_train, x2_train], y_train,
    batch_size      = 200,
    nb_epoch        = 100,
    show_accuracy   = True,
    validation_data = ([x1, x2], y)
)

pd.crosstab(
    np.reshape(model.predict([x1, x2]), (x1.shape[0],)) > .5,
    y    
)
