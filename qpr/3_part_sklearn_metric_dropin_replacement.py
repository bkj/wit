
# -- 
# ** Drop in replacement **
# Step2)  Modeling with untrained bag of words

from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer

data = df.groupby('schema').apply(lambda x: strat_pairs(x, n_match = 100, n_nonmatch = 50)).drop_duplicates()
docs = df.groupby('hash')['obj'].apply(lambda x: ' '.join(x))

vect  = CountVectorizer(ngram_range = (1, 2), stop_words = 'english', min_df = .01, max_df = .95)
tdmat = vect.fit_transform(docs)

sims  = 1 - cosine_distances(tdmat)

plt.plot(sorted(np.hstack(sims)))
plt.show()

eqv = np.vstack({tuple(x) for x in sims > .6})
eqv = [list(np.where(x)[0]) for x in eqv]
pprint(eqv)

# Ensure symmetric
assert(len(np.unique(np.hstack(eqv))) == len(np.hstack(eqv)))

eqv_map = []
counter = 0
for e in eqv:
    for h in e:
        eqv_map.append({"hash" : h, "field_id" : counter})
    
    counter += 1

dfm         = df.merge(pd.DataFrame(data = eqv_map))
fin         = dfm.groupby('id').apply(lambda x: pd.DataFrame(data = list(x['obj']), index = list(x['field_id']))).reset_index()
fin.columns = ['doc_id', 'field_id', 'obj']


out_store = pd.HDFStore('gun_domains.h5',complevel=9, complib='bzip2')

fin  = fin.sort('obj')
fino = fino.sort('obj')

pd.crosstab(np.array(fin['field_id']), np.array(fino['field_id']))

fin[fin['field_id'] == 3]['obj'].sample(10)






