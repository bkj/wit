from hashlib import md5
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 120)

# --
# Fuctions

def comp_css_path(elem):
    out = []
    q   = elem if elem.name else None
    for p in elem.parentGenerator():
        if q:
            idx = 1 + len(filter(lambda x: x.name == q.name, list(q.previous_siblings)))
            out.append({
                'sel'    : q.name + ':nth-of-type(' + str(idx) + ')',  # CSS Selector of exact element
                'wclass' : q.name + str(q.attrs.get('class', []))      # Equivalence class for modeling
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

df = pd.read_csv('../data/raw_gun.csv')

df['source'].value_counts()

# Format subset to work with
store = pd.HDFStore('gun_leaves_20151118_v2.h5',complevel=9, complib='bzip2')

usource = np.array(df.source.value_counts().index)
for source in usource:
    print '\n --- starting %s --- \n' % source
    if '/' + source in store.keys():
        print source + ' already exists!'
        continue
      
    print '+ making sub...'
    sub         = df[df['source'] == source]
    sub['soup'] = sub['content'].apply(BeautifulSoup)
    sub['id']   = range(sub.shape[0])
    
    # Extract data paths
    def get_data(inp):
        txt = filter(lambda x: len(x.strip()) > 0, inp[1].findAll(text = True))
        out = map(comp_css_path, txt)
        _   = map(lambda x: x.update({"id" : inp[0]}), out)
        return out
    
    print '+ getting data...'    
    all_df = map(get_data, zip(range(sub.shape[0]), sub['soup']))
    all_df = pd.concat(map(pd.DataFrame, all_df)).drop_duplicates()
    
    # Shortening hashes, if legal
    print '+ shortening hashes...'
    short_hash = all_df['hash'].apply(lambda x: x[0:5])
    if all_df['wclass'].unique().shape[0] == short_hash.unique().shape[0]:
        all_df['hash'] = short_hash
    
    # Subset to most general classes
    print '+ subsetting to general classes...'
    usel = all_df['wclass'].unique()
    for u in usel:
        keep   = ~ all_df.wclass.apply(lambda x: (u in x) and (u != x))
        all_df = all_df[keep]
    
    # Get the corresponding objects
    print '+ grabbing objects...'
    all_df['obj'] = all_df.apply(lambda x: sub.iloc[x['id']]['soup'].select(x['sel'])[0], 1)
    
    # Drop paths with no variation in obj
    print '+ dropping paths w/o variation'
    val_counts  = all_df.groupby('hash')['obj'].agg(lambda x: len(x.unique()))
    null_hashes = val_counts.index[val_counts == 1]
    all_df      = all_df[all_df.apply(lambda x: x['hash'] not in null_hashes, axis = 1)]
    
    # Save to file
    print '\n --- saving %s --- \n' % source
    all_df['obj'] = all_df['obj'].apply(str)
    store[source] = all_df

store.close()
