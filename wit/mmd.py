import numpy as np
from scipy.spatial.distance import pdist, squareform

# Private MMD helper
def _mmd(xx, yy, xy, wx, wy):
    if np.any(wx) or np.any(wy):
        xx = xx * np.outer(wx, wx)
        yy = yy * np.outer(wy, wy)
        xy = xy * np.outer(wx, wy)
        
        nx = wx.sum()
        ny = wy.sum()
    else:
        nx = x.shape[0]
        ny = y.shape[0]
        
    xx = (xx.sum() - nx) / (nx * (nx - 1))
    yy = (yy.sum() - ny) / (ny * (ny - 1))
    xy = xy.sum() / (nx * ny)
    return xx + yy - (2 * xy)


# Cosine similarity MMD (w/ optional weights)
# ** Assumes vectors are already normalized **
def cosine_mmd(x, y, wx = None, wy = None):
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
        
    return _mmd(xx, yy, xy, wx, wy)


# RBF (Radial basis function) MMD
def rbf_mmd(x, y, wx = None, wy = None, s = 1):
    X = np.vstack([x, y])
    
    paird = squareform(pdist(X, 'sqeuclidean'))
    paird = np.exp((- paird) / (s ** 2)) # This makes it a distance instead of a kernel
    
    xx = paird[:x.shape[0], :x.shape[0]]
    yy = paird[x.shape[0]:, x.shape[0]:]
    xy = paird[:x.shape[0], x.shape[0]:]
    
    return _mmd(xx, yy, xy, wx, wy)    
    # nx = x.shape[0]
    # ny = y.shape[0]
    
    # xx = xx.sum() / (nx ** 2)
    # yy = yy.sum() / (ny ** 2)
    # xy = xy.sum() / (nx * ny)
    
    # return -(xx + yy - (2 * xy))

x = np.random.uniform(0, 1, (1000, 1))
y = np.random.normal(0, 1, (1000, 1))
- rbf_mmd(x, y)


'''
# Example 1
# ---------

x = np.random.normal(0, 1, (10000, 1))
y = np.random.uniform(0, 1, (1000, 1))

cosine_mmd(x, y)

# Example 2
# (Weighted Cosine)
# -----------------

x    = np.random.normal(0, 1, (10, 1))
wx   = np.random.choice(range(1, 10), 10)
bigx = zip(wx, np.split(x, x.shape[0]))
bigx = np.vstack([np.tile(val, (rep, val.shape[1])) for rep,val in bigx])

y    = np.random.uniform(0, 1, (10, 1))
wy   = np.random.choice(range(1, 10), 10)
bigy = zip(wy, np.split(y, y.shape[0]))
bigy = np.vstack([np.tile(val, (rep, val.shape[1])) for rep,val in bigy])

cosine_mmd(bigx, bigy)

cosine_mmd(x, y, wx, wy)

'''