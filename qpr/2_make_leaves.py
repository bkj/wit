import json, base64, os
from hashlib import md5
from itertools import takewhile, izip
import numpy as np
import pandas as pd

from pprint import pprint
from bs4 import BeautifulSoup

from multiprocessing import Pool

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

# --
# Fuctions

# def allsame(x):
#     return len(set(x)) == 1

# def lcp(x):
#     return ''.join([i[0] for i in takewhile(allsame, izip(*x))])

def comp_css_path(elem):
    out = []
    q = None
    if elem.name:
        q = elem
    
    for p in elem.parentGenerator():
        if q:
            siblings = p.contents
            idx = 1 + len(filter(lambda x: x.name == q.name, list(q.previous_siblings)))
            
            out.append({
                'sel'    : q.name + ':nth-of-type(' + str(idx) + ')',  # Pointer to exact element
                'wclass' : q.name + str(q.attrs.get('class', []))      # Equiv class for modeling
            })
    
        q = p
    
    tmp = {
        'sel'    : ' > '.join(reversed([o['sel'] for o in out])),
        'wclass' : ' > '.join(reversed([o['wclass'] for o in out]))
    }
    
    tmp['hash'] = md5(tmp['wclass']).hexdigest()
    
    return tmp

# --
# Load data

df = pd.read_csv('raw_gun.csv')
df = df[~df['content'].apply(lambda x: 'postbottom' in x)]

df['source'].value_counts()

# Format subset to work with
store = pd.HDFStore('gun_leaves_noindex.h5',complevel=9, complib='bzip2')

for source in df['source'].unique():
    print source
    if '/' + source in store.keys():
        print source + ' already exists!'
        continue
    
    sub_df = df[df['source'] == source]
    
    tmpdata         = sub_df #.iloc[0:500]
    tmpdata['soup'] = tmpdata['content'].apply(BeautifulSoup)
    tmpdata['id']   = range(tmpdata.shape[0])
    
    # Extract data paths
    def get_data(i):
        if not i % 50:
            print i
        
        txt = filter(lambda x: len(x.strip()) > 0, tmpdata['soup'].iloc[i].findAll(text = True))
        out = pd.DataFrame(map(comp_css_path, txt)).drop_duplicates()
        out['id'] = i
        return out
    
    
    all_df = pd.concat(Pool().map(get_data, range(tmpdata.shape[0])))
    
    short_hash = all_df['hash'].apply(lambda x: x[0:5])
    if all_df['wclass'].unique().shape[0] == short_hash.unique().shape[0]:
        all_df['hash'] = short_hash
    
    # Subset to most general classes
    usel = all_df['wclass'].drop_duplicates()
    for u in usel:
        print u
        all_df = all_df[~all_df.apply(lambda x: (u in x['wclass']) and (u != x['wclass']), 1)]
    
    # Get the corresponding objects
    all_df['obj'] = all_df.apply(lambda x: tmpdata.iloc[x['id']]['soup'].select(x['sel'])[0], 1)
    
    # Drop paths with no variation in obj
    val_counts  = all_df.groupby('hash')['obj'].agg(lambda x: len(x.unique()))
    null_hashes = val_counts.index[val_counts == 1]
    all_df      = all_df[all_df.apply(lambda x: x['hash'] not in null_hashes, axis = 1)]
    
    # Save to file
    all_df_out = all_df.copy()
    all_df_out['obj'] = all_df_out['obj'].apply(str)
    store[source] = all_df_out


store.close()
