# what-is-this

Algorithms for string classification and string embeddings using 'weak' supervision, with eventual application to 'schema alignment'.

### Requirements

`Keras` fork from `https://github.com/bkj/keras`, which contains custom objective functions and regularizers

### Notes

Here are two ways that we could think about similarity of strings

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

- `wit/string-example.py` shows how to build a string classifier (ie `semantic`)
- `wit/simple-embedding-example.py` shows how to use the triplet loss function to learn a string embedding (ie `syntactic`)
- `wit/name-example.py` is a stub to remind myself to implement a string classifier that distinguishes between country of origin of name
