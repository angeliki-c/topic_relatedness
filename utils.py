from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
"""
    nparray_1   : shape  (n_samples, k_features)
    nparray_2   : shape (m_sample, k_features)
    
"""
def cosine_similarity(nparray_1, nparray_2):
    np.seterr(invalid = 'ignore')
    
    return np.divide(nparray_1, np.linalg.norm(nparray_1, axis = 1).reshape((nparray_1.shape[0],1)) )@ np.divide( nparray_2.T, np.linalg.norm(nparray_2.T, axis = 0))
    
    