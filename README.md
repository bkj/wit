# what-is-this

"Schema matching" algorithms

### Notes

Here are 2 ways that we can think about similarity of strings

- Syntactic : strings are similar, because they have similar structure
   - usernames : ben46 ~= frank123
   - subject_line : Re: good morning ~= Re: circling back
   
- Semantic : strings are similar, because of extrinsic information about the world
  - date : 2016-01-01 = Jan 1st 2016
  - country : AR = Argentina

Here's a way we can think about similarity of sets of strings:

- Distributional : sets have similar distributions
  - forum post_id  : (near?) unique key
  - forum username : may follow similar distributions across domains
  
- "Relational" : sets have similar relationships to other sets of strings
  - relationship (eg mutual information) between post_id and username may be similar across domains
