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
    "marauderairrifle.com"       : ['div', {'class' : 'post'}],
    "204ruger.com"               : ['div', {'class' : 'post'}],
    "www.rossi-rifleman.com"     : ['div', {'class' : 'post'}],
    "www.deerhuntingchat.com"    : ['div', {'class' : 'post'}],
    
    "www.combat-shotgun.com"     : ['div', {'class' : 'post'}],
    "www.newmexicoguntrader.net" : ['div', {'class' : 'post'}],
    "www.silencerforum.com"      : ['div', {'class' : 'post'}],
    
    "www.remingtonsociety.com"     : ['tr', {'class' : ['row1', 'row2']}], # Should merge like 1, 1, 2, 2, etc.  Done on line 70 now.
    "www.henryfirearms.org"        : ['tr', {'class' : ['row1', 'row2']}], # 
    "calicolightweaponsystems.com" : ['tr', {'class' : ['row1', 'row2']}],
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
df = df[~df['content'].apply(lambda x: 'postbottom' in x)].reset_index()
df.to_csv('raw_gun.csv')
