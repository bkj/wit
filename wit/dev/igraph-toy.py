import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import itertools
from igraph import *

from hashlib import md5

df         = pd.read_csv('data/simple-forum-dataset.csv')
df['node'] = df.apply(lambda x: md5(x['hash'] + x['obj']).hexdigest(), 1)

sub = df[df.hash.apply(lambda x: '1706c' in x)]

# --
idx        = list(sub.node.unique())
sub['gid'] = sub.node.apply(lambda x: idx.index(x))

# Make edge list
edges = sub.groupby('id').apply(lambda x: pd.DataFrame(list(itertools.combinations(x.gid, 2))))
edges = edges.reset_index()
edges = np.array(edges[[0, 1]], dtype = 'int')

g = Graph()
g.add_vertices(len(idx))
g.add_edges(map(tuple, edges))

# --

m = sub[['hash', 'gid']].drop_duplicates()
assert(list(m.gid) == sorted(list(m.gid)))
m['met'] = g.betweenness(directed = False)

m.head()

grp = m.groupby('hash').met.apply(np.mean)
grp

tmp = df[df.hash.isin(grp.index)]
tmp.obj.groupby(tmp.hash).apply(lambda x: x.head())


df.obj.value_counts().head()

plt.hist(list(m[m.hash == '459fb-1706c'].met.apply(lambda x: np.log(x + 1))))
plt.hist(list(m[m.hash == '2e98d-1706c'].met.apply(lambda x: np.log(1 + x))))
plt.show()

df[df.hash == '2e98d-1706c'].head()