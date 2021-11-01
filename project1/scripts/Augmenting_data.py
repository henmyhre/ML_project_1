# -*- coding: utf-8 -*-
"""
Functions used to augment the dataset
"""

import numpy as np

"""Data augmentation
The main function is add_features which augments the data.
It uses the functions:
    -add_log_terms, adds log(1+x) feature if all values in a column are bigger than -1
    -build_polynomial, adds a polynomial expansion of degree 0 up to the given degree
    -add_cross_terms, adds cross terms for all the columns"""

def add_features(data, degree = None, sqrt = True, log = True, cross_terms = True):
    """
    Adds following features to data set:
    -log of features by log(1+x)
    -sqrt of features
    -polynomial extension of 0 up to degree
    -cross terms of features
    input:
        data = ndarray of training/test dataset
        degree = int, highest degree of polynomial expansion to be added
        sqrt = boolean, if true sqrt feature will be added for all columns
        log =  boolean, if true log(1+x) feature will be added for all columns with all values greater than -1
        cross_terms = boolean, if true crossterms will be added between all columns
    output:
        output = ndarray of augmented dataset
    """ 
    #log
    if log:
        data = add_log_terms(data)
        output = data
    else:
        output = np.empty((data.shape[0],0))
    print(output.shape)
    #polynomial
    if degree is not None:
        output = np.c_[output, build_polynomial(data, degree)]
      
    # add sqrt
    if sqrt:
        output = np.c_[output, np.sqrt(np.abs(data))]
       
    
    if cross_terms:
        output = np.c_[output, add_cross_terms(data)]
            
    return output

def add_log_terms(data):
    """Adds log terms to data"""
    extended = data
    for column in data.T:
        if np.sum(column <= -1) == 0:
            extended = np.c_[extended, np.log(1+ column)]
        
    return extended

def build_polynomial(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    Extended = np.ones((x.shape[0],1))
    
    for j in range(2, degree+1):
        for i in range(x.shape[1]):
            Extended = np.c_[Extended, x[:,i]**j]
    
    return Extended


def add_cross_terms(data):
    """Adds cross terms between columns"""
    enriched_data = np.empty((data.shape[0], 0))
    
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            x1, x2 = data[:,i], data[:,j]
            if np.sum(x1 - x2) != 0:
                enriched_data = np.c_[enriched_data, x1*x2]
                
    return enriched_data      

