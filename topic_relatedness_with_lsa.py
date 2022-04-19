"""
    - read the data
    - data preprocessing
    - TF-IDF weighting scheme on the text
    - SVD
    - Addressing queries (queries.py)

"""

from topic_relatedness.stemmer import *	

from topic_relatedness.queries import *	

import pyspark.sql.functions as F

from pyspark.sql.types import StructType, StringType
from pyspark.ml.feature import CountVectorizer, IDF   


# they include information on the pages
dfac = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','page').load('hdfs://localhost:9000/user/data/atmospheric_sciences.xml')
dfdc = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','page').load('hdfs://localhost:9000/user/data/diseases_and_disorders.xml')
dfgc = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','page').load('hdfs://localhost:9000/user/data/geography.xml')

# consider the 'text' field from 'revision'
# these are the dataframes of the use case. Each contains information on the latest revision of the page's content. 
dfa = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','revision').load('hdfs://localhost:9000/user/data/atmospheric_sciences.xml')
dfd = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','revision').load('hdfs://localhost:9000/user/data/diseases_and_disorders.xml')
dfg = spark.read.format('xml').option('inferSchema','true').option('rootTag','mediawiki').option('rowTag','revision').load('hdfs://localhost:9000/user/data/geography.xml')

# it is good practice to cache it. I won't do it.
dfa = dfa.select(['id', 'text'])
dfd = dfd.select(['id', 'text'])
dfg = dfg.select(['id', 'text'])

# add the title of the page to the dataframe as a seperate column
con_list =  [(t1.title,t2.id) for t1, t2 in zip(dfac.select('title').collect(),dfa.select('id').collect())]
schema = StructType().add('title',StringType()).add('id',StringType())
# create a dataframe 'connector' for concatenating the 'title' column to the dataframe
dfa_con = spark.createDataFrame(con_list, schema, ['title','id'])
dfa = dfa.join(dfa_con, 'id')

con_list =  [(t1.title,t2.id) for t1, t2 in zip(dfdc.select('title').collect(),dfd.select('id').collect())]
# create a dataframe 'connector' for concatenating the 'title' column to the dataframe
dfd_con = spark.createDataFrame(con_list, schema, ['title','id'])
dfd = dfd.join(dfd_con, 'id')

con_list =  [(t1.title,t2.id) for t1, t2 in zip(dfgc.select('title').collect(),dfg.select('id').collect())]
# create a dataframe 'connector' for concatenating the 'title' column to the dataframe
dfg_con = spark.createDataFrame(con_list, schema, ['title','id'])
dfg = dfg.join(dfg_con, 'id')

# id is unique for all. Good.
assert dfa.count() == dfa.select('id').distinct().count()
assert dfd.count() == dfd.select('id').distinct().count()
assert dfg.count() == dfg.select('id').distinct().count()

dfa = dfa.withColumn('category',F.lit('atmospheric'))                   
dfd = dfd.withColumn('category',F.lit('disease'))
dfg = dfg.withColumn('category',F.lit('geography'))

# the 'id' column is not going to be unique now. Caution. 
df = dfa.union(dfd).union(dfg)


# replace 'id' with unique numbers, it will be useful to some specific queries
dfp = df.toPandas()
dfp['id'] = dfp.index
df = spark.createDataFrame(dfp, df.schema, df.columns)
df = df.withColumn('text', F.udf(lambda r : str(r.text._VALUE) + " " + str(r.title))(F.struct(df.text,df.title)))
df = df.cache()
df.where('id is null').count()     #    0

def feature_engineering(df):
    
    # nltk lemmatizer. gensim, spacy and Stanford coreNLP could work as well. 
    frdd = df.rdd.mapPartitions(lambda p : init_stemming(p))

    lem_df = frdd.toDF(['id','lem_toks','category'])
    lem_df = lem_df.cache()

    # for the tf transformation HashingTF or CountVectorizer may be used as well
    # tf with CountVectorizer
    # There are a couple of different options for configuring the CountVectorizer that result in different vocabularies.
    cv = CountVectorizer().setInputCol('lem_toks').setOutputCol('raw_features').setVocabSize(1000)
    cv_model= cv.fit(lem_df)
    tf_df = cv_model.transform(lem_df)
    tf_df = tf_df.cache()
    idf = IDF().setInputCol(cv.getOutputCol()).setOutputCol('features')
    idf_m = idf.fit(tf_df)
    tfidf_df = idf_m.transform(tf_df)
    tfidf_df = tfidf_df.cache()  
    vocabulary = cv_model.vocabulary
    
    
    return vocabulary, tfidf_df

vocabulary, tfidf_df = feature_engineering(df)

from  pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors

def svd(df, k):
    mod_rdd = df.rdd.map(lambda r : Vectors().fromML(r['features']))
    mod_rdd = mod_rdd.cache()  
    # returns a RowMatrix
    mat_rm =  RowMatrix(mod_rdd)              
                                                       
    svd_model = mat_rm.computeSVD(k,True)

    # returns a distributed RowMatrix
    # It correpsonds to a mapping between document space and concept space.
    U = svd_model.U                                       
    U_rdd = U.rows    
   
    # returns a DenseMatrix. It corresponds to a mapping between term space and concept space.
    V = svd_model.V                                              
    s = svd_model.s
    
    return U_rdd, s, V
    
U, s, V = svd(tfidf_df, 100)    

# Introducing some queries.

idx2voc = {i:el for i,el in enumerate(vocabulary)}
idx2doc = {i : r.title for i, r in enumerate(df.select('title').collect()) }
num_concepts = 100
num_terms = 200
num_docs = 50

most_important_terms_100 = most_important_terms(s = s, V = V, idx2voc = idx2voc, num_terms = num_terms, num_concepts = num_concepts)
print(f" The {num_terms} most important terms (assuming {num_concepts} most important concepts):\n {most_important_terms_100}")
most_important_docs_100 = most_important_docs(U = U,s = s, idx2doc = idx2doc, num_docs = num_docs, num_concepts = num_concepts)
print(f" The {num_docs} most important docs (assuming {num_concepts} most important concepts):\n {most_important_docs_100}")

terms_rel = terms_relevancy(s = s, V =V, idx2voc = idx2voc, num_terms = num_terms )
print(f"Relevancy between the first {num_terms} terms in the vocabulary :\n {terms_rel} ")
terms_most_rel = terms_relevancy(s = s, V =V, idx2voc = idx2voc, num_terms = num_terms, ascending =  False )
print(f"The relevancy between the first {num_terms} most relevant terms :\n {terms_most_rel} ")

term1 = idx2voc[56]
term2 = idx2voc[700]
score = relevance_between_terms(term1, term2, idx2voc , s = s, V = V)
print(f"Relevance between '{term1}' and '{term2}' is {score}")
num_docs = 100
docs_rel= docs_relevancy(U = U, s = s, idx2doc = idx2doc, num_docs = num_docs )
print(f"Relevancy between the first {num_docs} docs in the doc collection :\n {docs_rel} ")

docs_most_rel = docs_relevancy(s = s, U =U, idx2doc = idx2doc, num_docs = num_docs, ascending =  False )
print(f"Relevancy between the first {num_docs} most relevant docs :\n {docs_most_rel} ")
doc1 = idx2doc[76]
doc2 = idx2doc[79]
score_docs = relevance_between_docs(76, 79, idx2doc, U = U, s = s)
print(f"Relevance between doc '{doc1}' and doc '{doc2}' : {score_docs}")

term = idx2voc[60]
docs_to_term_rel = docs_relevancy_to_term(term, idx2voc, idx2doc,  U = U, s = s, V = V, num_docs = num_docs)
print(f"Documents with which term '{term}' has relevance : \n{docs_to_term_rel}")

# how much related is a list of random terms to the articles of the corpus.
# As the set of terms needs to pass through the whole processing pipeline and is regarded, 
# in the code, as a new doc in the corpus, a '.' should also be included in the list, in
# order the terms to be addressed as part of a sentence. This might be improved in future
# versions of the code.
terms = ['pollution', 'cancer', 'city', 'silicon', 'dust','.']
new_df = df
new_df.schema[-1].nullable = True    
new_doc = spark.createDataFrame([(new_df.agg(F.max('id').astype('long').alias('id')).collect()[0].id + 1, " ".join(terms), None, None)],new_df.schema, new_df.columns)
new_df = new_df.union(new_doc)
idx2doc = {r.id: r.title for r in new_df.select(['id','title']).collect()}
new_vocabulary, new_tfidf = feature_engineering(new_df)

U, s , V = svd(new_tfidf, k = 100)
doc = new_doc.select('id').collect()[0].id
res = doc_to_docs_relevancy( doc, U = U, s = s, V = None, idx2doc = idx2doc)
print(f"Terms {terms} relate to docs : \n {res}")