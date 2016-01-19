# wit

Algorithms for string classification and string embeddings using 'weak' supervision, with eventual application to 'schema alignment'.

### Method Overview

For schema alignment, basic idea is to:
   
   - learn an embedding of strings into dense N-dimensional vector representations s.t. instances of the same variable are closer than instances of other variables (recurrent neural networks)
   - align variables whose embedded distributions are "close" (solve assignment problem)


### Requirements

`Keras` fork from `https://github.com/bkj/keras`, which contains custom objective functions and regularizers

### Notes

Here are two ways that we could think about similarity of strings:

- `syntactic` : strings are similar, because they have similar structure
   - usernames : `ben46 is close to frank123`
   - subject_line : `'Re: good morning' is close to 'Re: circling back'`
   
- `semantic` : strings are similar, because of extrinsic information about the world
  - date : `'2016-01-01' is close to 'Jan 1st 2016'`
  - country : `'AR' is close to 'Argentina'`

and here are two ways we could think about similarity of sets of strings:

- `distributional` : sets have similar distributions
  - forum post_id  : (near?) unique key
  - forum username : may follow similar distributions across domains
  
- `relational` : sets have similar relationships to other sets of strings
  - relationship (eg mutual information) between post_id and username may be similar across domains

### Software

Prototype code for calculating `syntactic` and `semantic` similarity are included in this repo. 

#### Scripts
- `wit/examples/string-example.py` shows how to build a string classifier (ie `semantic`)
- `wit/examples/simple-embedding-example.py` shows how to use the triplet loss function to learn a string embedding (ie `syntactic`)
- `wit/examples/simple-alignment-example.py` -- splitting and re-aligning a simple dataset

#### Notebooks
- `wit/notebooks/address-matching.ipynb` -- trying to learn a good metric for addresses
- `wit/notebooks/simple-forum-notebook.py` -- aligning schemas of multiple forums at once

### More

See `https://github.com/phronesis-mnemosyne/census-schema-alignment` for some more concrete examples, developed during the January 2016 XDATA census hackathon.
