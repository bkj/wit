import numpy as np
from scipy.spatial.distance import pdist, squareform

# Cosine similarity MMD
# Accepts possible weights
def mmd(x, y, wx = None, wy = None):
    # Cosine similarity -- assuming already normalized.
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    
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


'''
# Example 1
# ---------

x = np.random.normal(0, 1, (10000, 1))
y = np.random.uniform(0, 1, (1000, 1))

mmd(x, y)

# Example 2
# ---------

x    = np.random.normal(0, 1, (10, 1))
wx   = np.random.choice(range(1, 10), 10)
bigx = zip(wx, np.split(x, x.shape[0]))
bigx = np.vstack([np.tile(val, (rep, val.shape[1])) for rep,val in bigx])

y    = np.random.uniform(0, 1, (10, 1))
wy   = np.random.choice(range(1, 10), 10)
bigy = zip(wy, np.split(y, y.shape[0]))
bigy = np.vstack([np.tile(val, (rep, val.shape[1])) for rep,val in bigy])

mmd(bigx, bigy)

mmd(x, y, wx, wy)

'''