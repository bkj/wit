import sys
import pandas as pd
import numpy as np

from time import time
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from .utils import text2feats

# --
# Helpers

def intuit_params(X, y):
    if type(X) == type([]):
        X = X[0]
    
    return {
        "n_classes"    : len(y),
        "num_features" : X.max() + 1,
        "max_len"      : X.shape[1],
    }

# --

class WitLoss:
    '''
        Special loss functions
    '''
    @staticmethod
    def dummy_loss(y_true, y_pred):
        return y_pred + (0 * y_true)
    
    @staticmethod
    def triplet_cosine(margin=0.3):
        def f(anc, pos, neg):
            posdist = 1 - (anc * pos).sum(axis=1, keepdims=True)
            negdist = 1 - (anc * neg).sum(axis=1, keepdims=True)
            return K.maximum(0, posdist - negdist + margin)
        return f


class WitClassifier:
    '''
        Base class for other classifiers
    '''
    model = None
    
    def __init__(self, num_features, max_len):
        self.num_features = num_features
        self.max_len = max_len
        
    def format_train(self, data):
        X = text2feats([x['value'] for x in data], self.max_len, self.num_features)
        y = pd.get_dummies([x['label'] for x in data])
        return X, np.array(y), np.array(y.columns)

    def format_test(self, data):
        return text2feats(data, self.max_len, self.num_features)


class StringClassifier(WitClassifier):
    '''
        Learn embedding given standard (example, label) pairs
    '''
    
    recurrent_size = 32
    dropout        = 0.5
    optimizer      = 'adam'
    
    def define_model(self):
        model = Sequential()
        model.add(Embedding(self.num_features, self.recurrent_size))
        model.add(LSTM(self.recurrent_size))
        model.add(Dropout(self.dropout))
        model.add(Dense(len(self.classes)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        
        self.model = model
    
    def fit(self, data, **kwargs):
        X, y, classes = self.format_train(data)
        self.classes = classes
        
        if not self.model:
            self.define_model()
        else:
            raise Warning("Make sure current labels match labels model was defined with.")
            
        fitist = self.model.fit(X, y, **kwargs)
        return fitist
    
    def predict(self, data, classes=True, **kwargs):
        X = self.format_test(data)
        preds = self.model.predict(X, **kwargs)
        if classes:
            return self.classes[preds.argmax(1)]
        else:
            return self.preds, classes


class TripletClassifier(WitClassifier):
    '''
        Learn embedding using (anc, pos, neg) triplets
        TODO : Make this adhere to the new API
    '''
    
    recurrent_size = 32
    dense_size     = 10
    optimizer      = 'adam'
    margin         = 0.3
    
    def define_model(self):        
        embedding = Embedding(self.num_features, self.recurrent_size)
        lstm      = LSTM(self.recurrent_size)
        dense     = Dense(self.dense_size)
        # ** Should we add dropout layer? **
        norm      = Activation(unit_norm)
        
        input_anc = Input(shape=(self.max_len, ), name='input_anc')
        input_pos = Input(shape=(self.max_len, ), name='input_pos')
        input_neg = Input(shape=(self.max_len, ), name='input_neg')
                
        proj_anc = norm(dense(lstm(embedding(input_anc))))
        proj_pos = norm(dense(lstm(embedding(input_pos))))
        proj_neg = norm(dense(lstm(embedding(input_neg))))
        
        triplet_merge = Merge(
            mode         = lambda tup: WitLoss.triplet_cosine(margin=self.margin)(*tup),
            output_shape = (None, 1),
        )([proj_anc, proj_pos, proj_neg])
        
        model = Model(input=[input_anc, input_pos, input_neg], output=triplet_merge)
        model.compile(loss=WitLoss.dummy_loss, optimizer=self.optimizer)
        
        self.input_layer = input_anc
        self.proj_layer  = proj_anc
        self.model = model
    
    def fit(self, anc, pos, neg, batch_size=128, nb_epoch=3):
        self.define_model()
        
        T = time()
        fitist = self.model.fit([anc, pos, neg], np.zeros(anc.shape[0]), nb_epoch=nb_epoch, batch_size=batch_size)
        print 'elapsed time :: %f' % (time() - T)
        return fitist
    
    def predict(self, data, batch_size=128, verbose=1):
        proj_model = Model(input=self.input_lay, output=self.proj_layer)
        proj_model.compile(loss=WitLoss.dummy_loss, optimizer=self.optimizer)
        return proj_model.predict(data, verbose=verbose, batch_size=batch_size)


