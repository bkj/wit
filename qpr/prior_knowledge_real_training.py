import pandas as pd
import numpy as np

from hashlib import md5

out_store = pd.HDFStore('gun_domains.h5',complevel=9, complib='bzip2')
source    = out_store.keys()[0]
df        = out_store[source]

field_id     = 5
sub          = df[df.field_id == field_id]
sub['class'] = sub.obj.apply(lambda x: md5(x[0:20]).hexdigest()[0:5])

# --

udocs = sub.doc_id.unique()

def make_real():
    out = []
    for u in udocs:
        tmp = sub[sub.doc_id == u]
        for i in range(tmp.shape[0]):
            for j in range(i + 1, tmp.shape[0]):
                out.append({
                    "match" : 0,
                    "obj1"  : tmp.obj.iloc[i],
                    "obj2"  : tmp.obj.iloc[j]
                })
    
    return pd.DataFrame(out)

def make_fake():
    s1 = sub.sample(1000).reset_index()
    s2 = sub.sample(1000).reset_index()
    
    fake = pd.concat([s1.obj.apply(lambda x: 1), s1.obj, s2.obj], axis = 1)
    fake.columns = ('match', 'obj1', 'obj2')
    
    return fake

train_data = pd.concat([make_real(), make_fake()])

sel   = np.random.uniform(0, 1, train_data.shape[0]) > .2

train = train_data.iloc[sel]
trn   = make_dataset(train, words = False)

valid = train_data.iloc[~sel]
val   = make_dataset(valid, words = False)



# --
# ... Train model ...
# --

def make_test():
    s1 = sub.sample(5000).reset_index()
    s2 = sub.sample(5000).reset_index()
    
    fake = pd.concat([0 + (s1['class'] == s2['class']), s1.obj, s2.obj], axis = 1)
    fake.columns = ('match', 'obj1', 'obj2')
    
    return fake


test = make_test()
tst  = make_dataset(test)

preds         = model.predict([tst['x1'], tst['x2']], verbose = 1)
preds.shape   = (preds.shape[0], )
test['preds'] = preds[:test.shape[0]]

plt.hist(test.preds, 100)
plt.show()

pd.crosstab(test.preds > 0.5, test.match)





