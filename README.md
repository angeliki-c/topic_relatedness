## topic_relatedness

### Discover and Exploit Topic Relatedness

    
 
Techniques followed
 
	Often, it is required in some applications to finding text content/articles that relate to a specific
	concept or topic, where using just standard search indexes has poor performance, as it is not feasible 
	all the words pertained to a topic to be included in a search index. Methods for retrieving documents 
	based on semantics, independent of whether specific keywords appear in the documents, exhibit better
	performance and one such method is LSA [1].
	
	LSA attributes a lower-diamensional representation to each term in the documents, computed using SVD.
	In SVD, the document-term matrix is represented by the product of three matrices, in which one matrix 
	represents the documents' space in respect with the concept space, one the concept space, indicating 
	the strength each concept has at the semantics level, and one the terms' space in respect with the
	concept space.
	
  There are multiple ways with which that task could be tackled. Another approach includes constructing
	the Vector Similarity Model (vsm) for the corpus, by creating the cooccurance matrix and then applying
	the tfidf scheme or an alternative one, such as ppmi or word2vec on the cooccurance matrix and 
	subsequently, instead of using lsa, another dimensionality reduction technique might be used such as 
	tsne.
	
	
  
Data set

	The corpus for this use case is composed of articles from Wikipedia, falling under the specific 
	categories:  atmospheric_sciences, diseases_and_disorders, geography, chemistry, biology and 
	physics. The documents were downloaded from Wikipedia's Export web site [2]. In this analysis, we
	are going to investigate semantic relatedness between articles specifically under the categories
    'atmospheric_sciences', 'diseases_and_disorders' and 'geography'. Of course we may try another 
	combination of categories to experiment with. In general, we may design any study or set of studies, 
  appropriately, on topic relatedness, involving different sets of articles and categories and try to
	answer intricate questions, such as is there any bias in the documents, which categories of articles
	tend to contain bias or is there any subtle connection between two concepts we don't know about yet,
	but the text written about each one higlights possible directions of connection between them? 

  Often, in clinical data there is an incremental number of cases reported for a particular type of 
	disease, connected with a specific geographical region, whereas in other regions the cases of that
	disease might be very low in number or absent. In addition atmospheric conditions increasingly seem
	to play role in human's health condition, therefore the emergence of some diseases may be connected 
	to the levels of pollution in a region or the atmospheric patterns may favour the healing, against 
	certain diseases or ward off the emergence of a disease. This research field has not been, yet, much
	explored and it is interesting to see, what connections there are between these concepts, at the moment,
	based on a literature's analysis.
	


Baseline
	
	It is interesting to set off a study that involves the comparison between the technique followed here,
	with other competent approaches, including using a baseline technique and baseline datasets. Though,
  this analysis doesn't aim to be exhaustive at the moment and a more thorough investigation is planned
  for the future.

	

Challenges

	A big part in data processing lifecycle consists of bringing raw data from multiple sources in a clean,
	most often tabular format, so that it can be further analyzed and used in learning certain ML algorithms.
	Unstructured text data presents a great challenge in bringing data to such a format, so that we could 
	extract greater value, from its subsequent analysis and processing.
	
	After applying an appropriate weighting scheme to the raw document-term matrix, a step which might involve 
	more than one stages of computation and may vary in general in diffrent applications of LSA, but typically 
	involving a Count Vectorizer or TFIDF transformation, SVD performs the low-rank approximation of the matrix
	and attributes a vector representation to each term, which captures the variance of each term within the
	context that the documents of the corpus defines. The resultant representation for each term, thus conveys 
	clues from the semantics each term most often adopts in text. Spark's ability to process an immense volume
	of text data efficiently, gives the opportunity for a variety of applications to emerge, taking advantage
	of the abudant text data, available in various kinds of forms.

  LSA doesn't take the order between words into account, neither negation of words.	
	
	The low-rank approximation of the document-term matrix contributes to an efficient representation of the data,
	saving resources, and accounts for challenges, such as synonymy between words, polysemy and noise in the data. 
    	
	Using the decomposition of the document-term matrix into three matrices and the representation of each document
	and each term in the context of the corpus, it is possible to explore the semantic content of our corpus,
	submitting a range of queries, such as the following, among others, which are implemented for our case study:
	- Which are the 100 most important concepts based on the corpus? Show me the first 200 terms that represent them.
	- Which are the 100 most important concepts based on the corpus? Show me the titles from the first 50 documents 
	  that represent them.
  - What similarity score do two specific terms from the vocabulary have?
    
  If we really want to extract the most value out of our text data, we should not constrain our exploration only in 
	looking up only the most similar components in our corpus, but invastigating the least connected entities, too or 
	entities belonging in specific percentiles of similarity score in our corpus. Also, submitting queries, using other
	metric functions (distance functions or others), might highlight information quite useful for our research.

    It is interesting to notice that in the results returned, after submitting the first query above, we see terms such
	  as: 
	  - 'lymphoma', which as other diseases its rate and symptoms are studied at a geographic level [4] and there are 
	    research reports claiming the connection of this disease with environmental factors, such as pollution [5]
      - 'hypertension', 'multimorbidity', 'cancer', 'mutation', 'drought', 'radiation' 	   
	  - exposome, which based on CDC "can be defined as the measure of all the exposures of an individual in a lifetime
    	and how those exposures relate to health" [6]
      - 'tccon', which stands for TCCON - Total Carbon Column Observing Network' 
	  - 'electric', which might be connected with the connection atmospheric electricity seems to have with biologic 
	     function [7]
    - 'noaa', which stands for National Oceanic and Atmospheric Administration. NOAA in a recent report states
   	   "Drought can harm food production and human health. Flooding can lead to disease spread and damages to ecosystems 
		   and infrastructure" [8]
	  - 'nbsp', 'ndash', 'valign', though terms like these show that there is some noise in the documents, which comes from
	     html and css language components, not interpreted by the browser, which appear on web pages of wikipedia.
    - also we identify terms that indicate possibly some existing bias in some texts in the corpus
    - a document with title 'Dysosteosclerosis' is among the articles related to the first 100 most important concepts, 
	    which refers to a rare disease of the bones. It is interesting that there are recent medical reports that suggest 
      air pollution's impact on bones' health [9] 
		
	These are only some of the observations one might extract out of the investigation on the topic relatedness in this 
	corpus, using lsa, that might prove useful in various research directions.
	
		

Evaluation

	For the evaluation of the similarity between vectors the cosine similarity has been used, but the code is
	configured to facilitate exploration with other metrics too.
	

 
Code

    topic_relatedness_with_lsa.py
   
    stemmer.py
       
	queries.py   
	
	utils.py
	   
   All can be run interactively with pyspark shell or by submitting e.g. exec(open("project/location/topic_relatedness/topic_relatedness_with_lsa.py").read()) 
   for an all at once execution. The code has been tested on a Spark standalone cluster. For the Spark setting,
   spark-3.1.2-bin-hadoop2.7 bundle has been used.
   The external python packages that are used in this implementation exist in the requirements.txt file. Install with: 
	   pip install -r project/location/topic_relatedness/requirements.txt
   This use case is inspired from the series of experiments presented in [3], though it deviates from it, in the
   programming language, the setting used and in the analysis followed.

   

References

	1. https://scikit-learn.org/stable/modules/decomposition.html#lsa
	2. https://en.wikipedia.org/wiki/Special:Export
  3. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills
  4. https://pubmed.ncbi.nlm.nih.gov/13874903/
  5. https://www.liebertpub.com/doi/10.1089/ees.2019.0241
  6. https://www.cdc.gov/niosh/topics/exposome/default.html#:~:text=The%20exposome%20can%20be%20defined,%2C%20diet%2C%20lifestyle%2C%20etc.
  7. https://link.springer.com/article/10.1007/s00484-020-02054-0
	8. https://www.noaa.gov/education/resource-collections/climate/climate-change-impacts
	9. https://www.sciencedaily.com/releases/2020/01/200103111726.htm
