import pandas as pd
import numpy as np

def preprocess(k, v):
    tmp         = pd.DataFrame(v)
    tmp['id']   = ['%s-%d' %(k, i) for i in range(tmp.shape[0])]
    tmp         = pd.melt(tmp, id_vars = ['id'])
    tmp['src']  = k
    tmp['hash'] = tmp.variable.apply(lambda x: k + '-' + x)
    tmp.rename(columns = {'value' : 'obj'}, inplace = True)
    
    tmp.obj = tmp.obj.apply(str)
    tmp     = tmp[tmp.obj.apply(lambda x: 'nan' != x)]
    tmp     = tmp[tmp.obj.apply(lambda x: 'None' != x)]
    
    return tmp


def make_distmat(preds, labs, levs):
    spreds = [preds[labs == i] for i in range(len(levs))]
    out = np.zeros( (len(levs), len(levs)) )
    for i in range(len(levs)):
        print i
        for j in range(i + 1, len(levs)):
            out[i, j] = fast_cosine_mmd(spreds[i], spreds[j])
    
    out          = out + out.T
    dist         = pd.DataFrame(out.copy())
    dist.columns = dist.index = levs
    return dist


def fast_distmat(preds, labs, levs):
    centers = np.vstack([preds[labs == i].mean(axis = 0) for i in range(len(levs))])
    sims    = centers.dot(centers.T)
    dg      = np.diag(sims)
    out     = np.vstack([dg[i] + dg for i in range(sims.shape[0])]) - 2 * sims
    
    dist         = pd.DataFrame(out.copy())
    dist.columns = dist.index = levs
    return dist


def fast_distmat_2(dpreds, dist_dist = False):
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
