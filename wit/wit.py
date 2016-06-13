from time import time

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
    
    def __init__(self, n_classes, num_features, max_len):
        self.n_classes = n_classes
        self.num_features = num_features
        self.max_len = max_len
        self.model = self.define_model()
    
    def predict(self, data, batch_size=128, verbose=1):
        return self.model.predict(data, verbose=verbose, batch_size=batch_size)


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
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        return model
    
    def fit(self, X, y, **kwargs):
        T = time()
        fitist = self.model.fit(X, y, **kwargs)
        print 'elapsed time :: %f' % (time() - T)
        return fitist


class TripletClassifier(WitClassifier):
    '''
        Learn embedding given (anc, pos, neg) triplets
    '''
    
    recurrent_size = 32
    dense_size     = 10
    optimizer      = 'adam'
    margin         = 0.3
    
    def define_model(self):        
        embedding = Embedding(self.num_features, self.recurrent_size)
        lstm      = LSTM(self.recurrent_size)
        dense     = Dense(self.dense_size)
        # ** Dropout? **
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
        
        return model
    
    def fit(self, anc, pos, neg, batch_size=128, nb_epoch=3):
        T = time()
        fitist = self.model.fit([anc, pos, neg], np.zeros(anc.shape[0]), nb_epoch=nb_epoch, batch_size=batch_size)
        print 'elapsed time :: %f' % (time() - T)
        return fitist
    
    def predict(self, data, batch_size=128, verbose=1):
        proj_model = Model(input=self.input_lay, output=self.proj_layer)
        proj_model.compile(loss=WitLoss.dummy_loss, optimizer=self.optimizer)
        return proj_model.predict(data, verbose=verbose, batch_size=batch_size)


