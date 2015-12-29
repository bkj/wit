import numpy as np
from scipy.spatial.distance import pdist, squareform

# Cosine mmd
def mmd(x, y, s = 1):
    X = np.vstack([x, y])
    
    # RBF
    # pd = squareform(pdist(X, 'sqeuclidean'))
    # pd = np.exp((- pd) / (s ** 2))
    
    # Cosine
    paird = np.dot(X, X.T)
    
    xx = paird[:x.shape[0], :x.shape[0]]
    yy = paird[x.shape[0]:, x.shape[0]:]
    xy = paird[:x.shape[0], x.shape[0]:]
    
    xx     = (xx.sum() - xx.shape[0]) / (xx.shape[0] * (xx.shape[0] - 1))
    yy     = (yy.sum() - yy.shape[0]) / (yy.shape[0] * (yy.shape[0] - 1))
    two_xy = (2 * xy.mean())
    
    return xx + yy - two_xy

# --
# EX

'''
x = np.random.normal(0, 1, 10000)
x = np.reshape(x, (x.shape[0],1))

y = np.random.uniform(0, 1, 1000)
y = np.reshape(y, (y.shape[0],1))

mmd(x, y)
'''