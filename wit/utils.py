import numpy as np

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def one_hot_character(text, n, lower=True):
    '''
        One hot encode string
        
        text - str - to encode
        n - int - max number of features
    '''
    if lower:
        text = text.lower()
    
    seq = list(text)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]

def text2feats(texts, maxlen, n=1000):
    '''
        One hot encode list of strings
        
        texts - *str - list of strs
        maxlen - int - truncate one hot encoding after maxlen characters
        n - int - max number of features
    '''
    X = np.zeros((len(texts), maxlen))
    for i,text in enumerate(texts):
        if len(t) > 0:
            X[i,-len(t):] = one_hot_character(text)
    
    return X