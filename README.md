# what-is-this

"Schema matching" algorithms

### Notes

Here are two ways that we could think about similarity of strings

- `syntactic` : strings are similar, because they have similar structure
   - usernames : ben46 ~= frank123
   - subject_line : Re: good morning ~= Re: circling back
   
- `semantic` : strings are similar, because of extrinsic information about the world
  - date : 2016-01-01 = Jan 1st 2016
  - country : AR = Argentina

and here are two ways we could think about similarity of sets of strings:

- `distributional` : sets have similar distributions
  - forum post_id  : (near?) unique key
  - forum username : may follow similar distributions across domains
  
- `relational` : sets have similar relationships to other sets of strings
  - relationship (eg mutual information) between post_id and username may be similar across domains

### Software

Prototype code for calculating `syntactic` and `semantic` similarity are included in this repo, under `wit`
