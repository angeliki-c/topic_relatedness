import pandas as pd
import numpy as np

from topic_relatedness.utils import *


"""    Queries    """

""""
    e.g 100 MOST IMPORTANT TERMS                                        
    Note: The terms that are highest in the rank do not provide much information on the semantic basis of their relatedness.
          Subsequent terms provide more information on this basis. The same holds for other similar to this queries below.
          
"""

def most_important_terms(U = None, s = None, V = None, idx2voc = None, num_terms = None, num_concepts = None, similarity_func = cosine_similarity):
    if num_concepts > len(s):
        raise ValueError("""The number of concepts corresponds to the selected dimension for the representation of the terms'
                            and docs' space. This should be <= the dimension of s. """)
                            
    Vr = V.toArray()[:,:num_concepts]
    # s is sorted. num_concepts most important concepts
    i_con = s[:num_concepts]
    si = np.eye(num_concepts) * i_con	
								
    # each instance (term) is represented by a size-num_concepts vector in the concept space
    sel_vecs = Vr @ si   
    score = np.linalg.norm(sel_vecs, axis = 1)
    #score = np.max(sel_vecs, axis = 1)
    scorep = pd.DataFrame(score, columns = ['value'])
    i_terms_idx = scorep.sort_values(by = 'value', ascending = False).index[:num_terms]
   
    i_terms = [idx2voc[idx] for idx in i_terms_idx]       
	
    return i_terms    
					

                    
# e.g 100 MOST_IMPORTANT_DOCS

def most_important_docs(U = None,s = None, V = None, idx2doc = None, num_docs = None, num_concepts = None):
    if num_concepts > len(s):
        raise ValueError("""The number of concepts corresponds to the selected dimension for the representation of the terms'
                            and docs' space. This should be <= the dimension of s. """)                             
    Ur = np.array(U.collect())[:,:num_concepts]                                                               
    # s is sorted. num_concepts most important concepts
    i_con = s[:num_concepts]
    si = np.eye(num_concepts) * i_con	
    # each instance (doc) is represented by a size-num_concepts vector in the concept space
    sel_docs = Ur @ si   
    score = np.linalg.norm(sel_docs, axis = 1)
    #score = np.max(sel_docs, axis = 1)
    scorep = pd.DataFrame(score, columns = ['value'])
    i_docs_idx = scorep.sort_values(by = 'value', ascending = False).index[:num_docs]
    imp_docs = [idx2doc[idx] for idx in i_docs_idx]
	
    return imp_docs   
    
                      

# TERMS RELEVANCY

def terms_relevancy(U = None, s = None, V = None, idx2voc = None, num_terms = None , ascending = None):
    
    # various similarity/disimilarity metrics can be used, such as : cosine similarity, jacard distance, (or scipy.spatial.distance with any metric you like)
    Vr = V.toArray()                                  					
    # s is sorted. num_concepts most important concepts
    si = np.eye(Vr.shape[1]) * s					
    # each instance (term) is represented by a size-num_concepts vector in the concept space
    sel_vecs = Vr @ si   
    cos_sim_mx = cosine_similarity(sel_vecs, sel_vecs)
    #cos_sim_mx = (sel_vecs/(np.linalg.norm(sel_vecs, axis = 1).reshape((sel_vecs.shape[0],1))) )@ (sel_vecs.T/(np.linalg.norm(sel_vecs.T, axis = 0)))
    # or osine_similarity from sklearn may be used instead
    #cos_sim_mx = cosine_similarity(sel_vecs, sel_vecs)
    
    if ascending == None:
        terms_relevance = pd.DataFrame(cos_sim_mx[:num_terms,:num_terms], index = list(idx2voc.values())[:num_terms], columns = list(idx2voc.values())[:num_terms]).fillna(0)
    elif ascending == False:
       
        arg_sim = np.argsort(cos_sim_mx.flatten()).tolist()[::-1]
        count = 1
        sim_shape = cos_sim_mx.shape[0]
        terms_rel_dict = dict()
        for i in range(len(arg_sim)):
            row = arg_sim[i] // sim_shape
            col = arg_sim[i] % sim_shape
            if row < col:
                if count == 1:
                    terms_rel_dict.update({'term1':[idx2voc[row]],'term2':[idx2voc[col]],'score': [cos_sim_mx[row,col]]})
                else:
                    terms_rel_dict['term1'].append(idx2voc[row])
                    terms_rel_dict['term2'].append(idx2voc[col])
                    terms_rel_dict['score'].append(cos_sim_mx[row,col])
                count += 1
                
                if count == num_terms:
                    break
        terms_relevance = pd.DataFrame(terms_rel_dict).fillna(0)
         
    elif ascending == True:
        # returns the least similar terms
        arg_sim = np.argsort(cos_sim_mx.flatten())
        count = 1
        sim_shape = cos_sim_mx.shape[0]
        terms_rel_dict = dict()
        for i in range(arg_sim.shape[0]):
            row = arg_sim[i] // sim_shape
            col = arg_sim[i] % sim_shape
            if row < col:
                if count == 1:
                    terms_rel_dict.update({'term1':[idx2voc[row]],'term2':[idx2voc[col]],'score': [cos_sim_mx[row,col]]})
                else:
                    terms_rel_dict['term1'].append(idx2voc[row])
                    terms_rel_dict['term2'].append(idx2voc[col])
                    terms_rel_dict['score'].append(cos_sim_mx[row,col])
                count += 1
                if count == num_terms:
                    break
        terms_relevance = pd.DataFrame(terms_rel_dict).fillna(0) 
    return terms_relevance
    
															    
# Some of the important concepts may be ambiguous, but the rest may correspond to meaningful categories.
# This analysis is usuful in identifying possible characteristics that connect terms and documents that are latent. (Bias?!)

def relevance_between_terms(term1, term2, idx2voc, U = None, s = None, V = None):		
    #  How relevant is a term to another term?		
    voc2idx = {v:k for k, v in idx2voc.items()}    
    idx1 = voc2idx[term1]
    idx2 = voc2idx[term2]
    Vr = V.toArray()[[idx1, idx2],:]                                  					
    si = np.eye(Vr.shape[1]) * s					
    sel_vecs = Vr @ si   
    #cos_sim_mx = (sel_vecs/(np.linalg.norm(sel_vecs, axis = 1).reshape((sel_vecs.shape[0],1))) )@ (sel_vecs.T/(np.linalg.norm(sel_vecs.T, axis = 0)))
    cos_sim_mx = cosine_similarity(sel_vecs, sel_vecs)
    return  cos_sim_mx[0,1]
  
  
def doc_to_docs_relevancy(doc, U = None, s = None, V = None, idx2doc = None):
    # various similarity/disimilarity metrics can be used, such as : cosine similarity, jacard distance, (or scipy.spatial.distance with any metric you like)
    Uall = np.array(U.collect())
    Ur = Uall[doc,:].reshape((1, len(s)))
    
    Urest = np.delete(Uall,doc,axis = 0)
    # s is sorted. num_concepts most important concepts
    si = np.diag(s)
    # each instance (doc) is represented by a size-num_concepts vector in the concept space
    sel_vecs = Ur @ si   
    rest_vecs = Urest @ si
    cos_sim_mx = cosine_similarity(sel_vecs, rest_vecs)
    #cos_sim_mx = (sel_vec/(np.linalg.norm(sel_vec)) )@ ((rest_vecs/(np.linalg.norm(rest_vecs, axis = 1).reshape((rest_vecs.shape[0],1)))).T)
    #cos_sim_mx = cosine_similarity(sel_vec,rest_vecs)
    index = np.delete(np.arange(len(idx2doc)), doc)
    docs_rel = pd.DataFrame(cos_sim_mx.squeeze(0),columns= ['score'])
    docs_rel['title'] = pd.Series([idx2doc[i] for i in index])
    docs_rel = docs_rel.sort_values(by = 'score', ascending = False).fillna(0)
    
    return docs_rel
    
    

def docs_relevancy(U = None, s = None, V = None, idx2doc = None, num_docs = None, ascending  = None ):
    # various similarity/disimilarity metrics can be used, such as : cosine similarity, jacard distance, (or scipy.spatial.distance with any metric you like)
    Ur = np.array(U.collect())
    # s is sorted. num_concepts most important concepts
    si = np.eye(Ur.shape[1]) * s					
    # each instance (doc) is represented by a size-num_concepts vector in the concept space
    sel_vecs = Ur @ si   
    #cos_sim_mx = (sel_vecs/(np.linalg.norm(sel_vecs, axis = 1).reshape((sel_vecs.shape[0],1))) )@ (sel_vecs.T/(np.linalg.norm(sel_vecs.T, axis = 0)))
    cos_sim_mx = cosine_similarity(sel_vecs, sel_vecs)
    doc_titles = list(idx2doc.values())
    if ascending == None:
        docs_relevance = pd.DataFrame(cos_sim_mx[:num_docs,:num_docs], index = doc_titles[:num_docs], columns = doc_titles[:num_docs]).fillna(0)
    elif ascending == False:
       
        arg_sim = np.argsort(cos_sim_mx.flatten()).tolist()[::-1]
        count = 1
        sim_shape = cos_sim_mx.shape[0]
        docs_rel_dict = dict()
        for i in range(len(arg_sim)):
            row = arg_sim[i] // sim_shape
            col = arg_sim[i] % sim_shape
            if row < col:
                if count == 1:
                    docs_rel_dict.update({'doc1':[idx2doc[row]],'doc2':[idx2doc[col]],'score': [cos_sim_mx[row,col]]})
                else:
                    docs_rel_dict['doc1'].append(idx2doc[row])
                    docs_rel_dict['doc2'].append(idx2doc[col])
                    docs_rel_dict['score'].append(cos_sim_mx[row,col])
                count += 1
                if count == num_docs:
                    break
        docs_relevance = pd.DataFrame(docs_rel_dict).fillna(0)
         
    elif ascending == True :
        
        arg_sim = np.argsort(cos_sim_mx.flatten())
        count = 1
        sim_shape = cos_sim_mx.shape[0]
        docs_rel_dict = dict()
        for i in range(arg_sim.shape[0]):
            row = arg_sim[i] // sim_shape
            col = arg_sim[i] % sim_shape
            if row < col:
                if count == 1:
                    docs_rel_dict.update({'doc1':[idx2doc[row]],'doc2':[idx2doc[col]],'score': [cos_sim_mx[row,col]]})
                else:
                    docs_rel_dict['doc1'].append(idx2doc[row])
                    docs_rel_dict['doc2'].append(idx2doc[col])
                    docs_rel_dict['score'].append(cos_sim_mx[row,col])
                count += 1
                if count == num_docs:
                    break
        docs_relevance = pd.DataFrame(docs_rel_dict).fillna(0)
         
    return docs_relevance
  
    
def relevance_between_docs(doc1, doc2, idx2doc, U = None, s = None, V = None):
    
    # various similarity/disimilarity metrics can be used, such as : cosine similarity, jacard distance, (or scipy.spatial.distance with any metric you like)
    Ur = np.array(U.collect())[[doc1, doc2],:]
    # s is sorted. num_concepts most important concepts
    si = np.eye(Ur.shape[1]) * s					
    # each instance (doc) is represented by a size-num_concepts vector in the concept space
    sel_vecs = Ur @ si   
    #cos_sim_mx = (sel_vecs/(np.linalg.norm(sel_vecs, axis = 1).reshape((sel_vecs.shape[0],1))) )@ (sel_vecs.T/(np.linalg.norm(sel_vecs.T, axis = 0)))
    cos_sim_mx = cosine_similarity(sel_vecs, sel_vecs)
   
    return cos_sim_mx[0,1]
    

    
def docs_relevancy_to_term(term, idx2voc, idx2doc,  U = None, s = None, V = None, num_docs = None):    
    voc2idx = {v:k for k,v in idx2voc.items()}
    idx = voc2idx[term]
    Vr = V.toArray()[idx,:].reshape((1, len(s))) 
    Ur = np.array(U.collect())
    si = (np.eye(Vr.shape[1]) * s)/ np.linalg.norm(s, ord = np.inf)
    sel_vecs = Ur @ si @ Vr.T
    scores = pd.DataFrame(sel_vecs, index = list(idx2doc.values()), columns = ['value'])
    first_most_relevant_docs = scores.sort_values(by = 'value',ascending = False).iloc[:num_docs,:]
    
    return first_most_relevant_docs


  