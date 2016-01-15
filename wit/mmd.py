import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd

# Private MMD helper
def _mmd(xx, yy, xy, wx, wy):
    if np.any(wx) or np.any(wy):
        xx = xx * np.outer(wx, wx)
        yy = yy * np.outer(wy, wy)
        xy = xy * np.outer(wx, wy)
        
        nx = wx.sum()
        ny = wy.sum()
    else:
        nx = xx.shape[0]
        ny = yy.shape[0]
        
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
    paird = np.exp((- paird) / (s ** 2))
    
    xx = paird[:x.shape[0], :x.shape[0]]
    yy = paird[x.shape[0]:, x.shape[0]:]
    xy = paird[:x.shape[0], x.shape[0]:]
    
    return _mmd(xx, yy, xy, wx, wy)    


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

# --
# Faster approximate methods
# - To do -- this is biased, need to correct -
# def quickdotmean(a, b, n_components = 1, prec = False):
#     if not prec:
#         sau, sad, savt = randomized_svd(a, n_components=n_components, n_iter=5, random_state=None)
#         sbu, sbd, sbvt = randomized_svd(b, n_components=n_components, n_iter=5, random_state=None)
#     else:
#         sau, sad, savt = a
#         sbu, sbd, sbvt = b
    
#     m = np.diag(sad).dot(savt).dot(sbvt.T).dot(np.diag(sbd))
#     return sau.mean(axis = 0).dot(m).dot(sbu.mean(axis = 0))

# def approx_cosine_mmd(a, b, n_components = 1, prec = False):
#     aa = quickdotmean(a, a, n_components, prec)
#     bb = quickdotmean(b, b, n_components, prec)
#     ab = quickdotmean(a, b, n_components, prec)
#     return aa + bb - (2 * ab)


def fastdotmean(a, b = None):
    if not np.any(b):
        tot = (a.sum(axis = 0) ** 2).sum()
        na  = a.shape[0]
        return (tot - na) / (na * (na - 1))
    else:
        tot = (a.sum(axis = 0) * b.sum(axis = 0)).sum()
        na  = a.shape[0]
        nb  = b.shape[0]
        return tot / (na * nb)


def fast_cosine_mmd(a, b, n_components = 1, prec = False):
    aa = fastdotmean(a, a)
    bb = fastdotmean(b, b)
    ab = fastdotmean(a, b)
    return aa + bb - (2 * ab)

# --

def dfdist(dpreds, dist_dist = False):
    preds_  = dpreds[dpreds.columns[:-1]]
    ulab    = list(dpreds.lab.unique())
    centers = np.vstack([preds_[dpreds.lab == i].mean(axis = 0) for i in ulab])
    
    sims = centers.dot(centers.T)    
    dg   = np.diag(sims)
    out  = np.vstack([dg[i] + dg for i in range(sims.shape[0])]) - 2 * sims
    
    if dist_dist:
        out = squareform(pdist(out.copy()))
    
    dist         = pd.DataFrame(out)
    dist.columns = dist.index = ulab
    return dist
