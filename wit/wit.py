# *** Only works with Theano ***

import re
import numpy as np
import pandas as pd

from time import time
from string import punctuation

import keras.backend as K
from theano import tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.preprocessing.text import one_hot

PUNCT_REGEX = re.compile(r'([\s{}]+)'.format(re.escape(punctuation)))

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

# --
# Keras Functions

def triplet_cosine(y_true, y_pred, margin=0.3):
    posdist = 1 - K.sum(y_pred[0::3] * y_pred[1::3], axis=-1)
    negdist = 1 - K.sum(y_pred[0::3] * y_pred[2::3], axis=-1)
    loss    = K.maximum(0, posdist - negdist + margin) - (y_true[0] * 0)
    return T.extra_ops.repeat(x, n)(loss, 3)

def unit_norm(x):
    return x / K.sqrt(K.sum(x ** 2, axis=-1, keepdims = True))


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


def one_hot_custom(x, n, filters = ''):
    return [abs(hash(t)) % (n - 1) + 1 for t in x.split(' ') if t not in filters]


class KerasFormatter:
    
    def __init__(self, num_features = 1000, max_len = 150):
        self.num_features = num_features
        self.max_len      = max_len
    
    def format(self, data, xfields, yfield, words=False, custom=False):
        if not isinstance(xfields, list):
            raise Exception('xfields must be a list')
        
        if len(xfields) > 2:
            raise Exception('illegal number of xfields')
        
        levs = sorted(list(data[yfield].unique()))
        xs   = [self._format_x(data[xf], words=words, custom=custom) for xf in xfields]
        y    = self._format_y(data[yfield], levs)
        return ({'x' : xs, 'y' : y}, levs)
        
    def _format_x(self, z, words=False, custom=False):
        oh = one_hot_custom if custom else one_hot
        return sequence.pad_sequences(
            [
                oh(string_explode(x, words=words), self.num_features, filters = '') for x in z
            ], 
            maxlen = self.max_len
        )
    
    def _format_y(self, z, levs):
        return np_utils.to_categorical([levs.index(x) for x in z])


# --
# Neural network string classifier

class WitClassifier:
    model = None
    
    def __init__(self, train, levs, model = None, opts = {}):        
        self.train = train    
        self.levs  = levs
        
        self.intuit_params()
        
        for k,v in opts.iteritems():
            setattr(self, k, v)
        
        if model:
            self.model = model
        else:
            print '--- compiling model --- \n'
            self.model = self.compile()
    
    def intuit_params(self):            
        self.n_classes    = len(self.levs)
        self.num_features = self.train['x'][0].max() + 1
        self.max_len      = self.train['x'][0].shape[1]
    
    def predict(self, data, batch_size = 128, verbose = 1):
        return self.model.predict(data, verbose = verbose, batch_size = batch_size)


class StringClassifier(WitClassifier):
    
    recurrent_size = 32
    dropout        = 0.5
    
    def compile(self):
        model = Sequential()
        model.add(Embedding(self.num_features, self.recurrent_size))
        model.add(LSTM(self.recurrent_size))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")
        return model
    
    def fit(self, **kwargs):
        if len(self.train['x']) > 1:
            raise Exception('train[x] has more than one entry -- stopping')
        
        kwargs['X']                = self.train['x'][0]
        kwargs['y']                = self.train['y']
        kwargs['validation_split'] = 0.2
        kwargs['show_accuracy']    = True
        
        return self.model.fit(**kwargs)
    
    def classify_string(self, string):
        num_features = 1000 # Why am I setting this?
        max_len      = 50   # Why am I setting this?
        z            = np.array(one_hot(string_explode(string), num_features, filters = ''))
        z.shape      = (1, z.shape[0])
        z            = sequence.pad_sequences(z, max_len)
        return self.levs[self.model.predict(z, batch_size = 1).argmax()]


class TripletClassifier(WitClassifier):
    
    recurrent_size = 32
    dense_size     = 10
    
    def compile(self):
        print '\t %d \t recurrent_size' % self.dense_size
        print '\t %d \t dense_size'     % self.recurrent_size
        
        model = Sequential()
        model.add(Embedding(self.num_features, self.recurrent_size))
        model.add(LSTM(self.recurrent_size))
        model.add(Dense(self.dense_size))
        model.add(Activation(unit_norm))
        model.compile(loss = triplet_cosine, optimizer = 'rmsprop')
        return model
    
    def fit(self, batch_size = 100, nb_epoch = 3):
        T = time()
        for n in range(nb_epoch):
            print 'n :: %d' % n
            
            x  = self.train['x'][0]
            
            ms = modsel(x.shape[0], N = 3)
            _  = self.model.fit(
                x[ms],
                x[ms], 
                nb_epoch   = 1,
                batch_size = 3 * batch_size,
                shuffle    = False
            )
                
        print 'elapsed time :: %f' % (time() - T) 
        return True

# --
# Helpers

def string_explode(x, words=False):
    if not words:
        tmp = ' '.join(list(unicode(x))[::-1]).strip()
    elif words:
        tmp = re.sub(PUNCT_REGEX, ' \\1 ', unicode(x))
        tmp = re.sub(' +', ' ', tmp)
    
    return tmp

# Helper function for making testing data
def strat_pairs(df, n_match=100, n_nonmatch=10, hash_id='hash'):
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


def print_eqv(eqv, df, path = 'obj'):
    for e in eqv:
        print bcolors.WARNING + '\n --- \n'
        print e
        print '\n' + bcolors.ENDC
        for h in e:
            print bcolors.OKGREEN + h + '\t(%d rows)' % df[df.hash == h].shape[0] + bcolors.ENDC
            print df[df.hash == h][path].sample(5, replace = True)


def modsel(S, N = 3):
    '''
        Permute order while keeping blocks of N stable
    '''
    n_samp = S / N
    sel    = N * np.random.choice(range(n_samp), n_samp)
    sels   = np.vstack([a + sel for a in range(N)]).T
    return np.reshape(sels, (n_samp * N))


def make_triplet_train(df, N = 200):
    print '\n'
    out     = []
    uhash   = df.hash.unique()
    
    counter = 0
    for ind, uh in enumerate(uhash):
        print '\t' + bcolors.OKGREEN + str(ind) + '\t' + uh + bcolors.ENDC
        
        pos = df[df.hash == uh].sample(N * 2, replace = True)
        neg = df[(df.hash != uh) & df.id.isin(pos.id.unique())].sample(N, replace = True)
        
        pos['doc'] = uh
        neg['doc'] = uh
        
        pos['role'] = 'pos'
        neg['role'] = 'neg'
        
        for i in range(N):
            anc_ = pos.iloc[i].to_dict()
            pos_ = pos.iloc[N + i].to_dict()
            
            # neg_ = neg[neg.id == anc_['id']].sample(1).iloc[0].to_dict()
            neg_ = neg.iloc[i].to_dict()
            
            anc_['ex'] = counter
            pos_['ex'] = counter
            neg_['ex'] = counter
            
            anc_['role'] = 'anc'

            out += [ anc_, pos_, neg_ ]
            
            counter += 1
    
    print '\n'
    return pd.DataFrame(out)
