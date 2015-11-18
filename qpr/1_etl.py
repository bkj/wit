import json, base64, os

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

# --
# Loading
path_to_data = '../raw_data/'
paths = filter(lambda x: ('crdownload' not in x) and ('jl' in x), os.listdir(path_to_data))

data = []
for p in paths:
    print 'reading %s' % p
    with open(path_to_data + p) as infile:
        x = infile.read().split('\n')
        x = filter(lambda x: x != '', x)
        x = map(json.loads, x)
        data += x

# --
# Extract post items

# Get post pages
d = filter(lambda x: x.get('_type', False) == 'CcaItem', data)
d = filter(lambda x: 'viewtopic' in x['url'], d)

# Get individual posts
paths = {
    "www.arguntrader.com"        : ['div', {'class' : 'post'}],
    "www.remingtonsociety.com"   : ['tr', {'class' : ['row1', 'row2']}], # Should merge like 1, 1, 2, 2, etc
    "www.henryfirearms.org"      : ['tr', {'class' : ['row1', 'row2']}], # ^^
    "www.combat-shotgun.com"     : ['div', {'class' : 'post'}],
    "www.newmexicoguntrader.net" : ['div', {'class' : 'post'}],
    "www.silencerforum.com"      : ['div', {'class' : 'post'}],
}

out = {}
for k, v in paths.iteritems():
    out[k] = []
    xs = filter(lambda x: x['response']['server']['hostname'] == k, d)
    xs = np.random.choice(xs, 500)
    
    counter = 0
    for x in xs:
        soup    = BeautifulSoup(base64.standard_b64decode(x['response']['body']))
        out[k] += soup.findAll(paths[k][0], paths[k][1])
        
        print 'counter: %d \t len: %d' % (counter, len(out[k]))
        counter += 1


df = []
for k,v in out.iteritems():
    for vv in v:
        df.append({
            "source"  : k,
            "content" : str(vv)
        })

df = pd.DataFrame(df)
df.to_csv('csvs/raw_gun.csv')

# -----------------------------------------
# -- Everything below this is deprecated --

def css_path(elem, show_pos = True):
    out = []
    q = None
    if elem.name:
        q = elem
    
    for p in elem.parentGenerator():
        if q:
            siblings = p.contents
            idx = 1 + len(filter(lambda x: x.name == q.name, list(q.previous_siblings)))
            if show_pos:
                out.append(q.name + ':nth-of-type(' + str(idx) + ')' + str(q.attrs.get('class', [])))
            else:
                out.append(q.name + str(q.attrs.get('class', [])))
        
        q = p
    
    return ' > '.join(reversed(out))

def depth_trim_soup(d, DEPTH = 5, remove_attrs = True, remove_content = True):
    soup      = BeautifulSoup(str(d))
    fa        = soup.findAll()
    to_remove = filter(lambda x: len(list(x.parents)) > DEPTH, fa)
    _ = [x.extract() for x in to_remove]
    
    if remove_attrs:
        for e in soup.findAll():
            # e.attrs = {}
            e.attrs['id']   = None
            e.attrs['href'] = None
            
    
    if remove_content:
        for e in soup.findAll(text = True):
            e.extract()
            
    return soup


def soupstring(x):
    return ''.join(str(x).split('\n'))

# Sample of content

df = pd.read_csv('csvs/raw_gun.csv')
df['source'].value_counts()

sub_df  = df[df['source'] == 'www.silencerforum.com']
tmpdata = sub_df['content'].iloc[0:500]

# Compute truncated DOM -- need to come up with scheme for
# finding DEPTH -- not exactly sure what to do
#
# - Maybe don't want to go past nodes that have text in them
# - Maybe want to remove <strong>, <br>, <b>, <i>, etc.
soups    = map(lambda x: depth_trim_soup(x, DEPTH = 6), tmpdata)
sstrings = np.array(map(soupstring, soups))
ustrings = pd.Series(sstrings).value_counts()
ustrings.shape

sstrings[0]

# For a single type of DOM, get paths for leaf nodes
def paths_from_pattern(pattern):
    fa        = BeautifulSoup(pattern).findAll()
    fa        = filter(lambda x: len(x.contents) == 0, fa)
    css_paths = map(css_path, fa)
    return css_paths

def extract_content_by_path(css_paths, instances):
    out = []
    for i, inst in enumerate(instances):
        print i
        out.append(map(lambda path: BeautifulSoup(inst).select(path)[0], css_paths))
    
    return out

all_df = []
for K in range(len(ustrings)):
    print K
    css_paths = paths_from_pattern(ustrings.index[K])
    instances = tmpdata[sstrings == ustrings.index[K]]
    content   = extract_content_by_path(css_paths, instances)
    tmpdf     = pd.DataFrame(content)
    all_df.append(tmpdf)


list(all_df[0][5].head())
