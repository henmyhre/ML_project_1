# -*- coding: utf-8 -*-
"""
Functions used for data loading and preprocessing of the data

"""
import numpy as np

"""Data loading"""

def load_results(path_dataset):
    """load results.
    Sets s to one and b to zero """
    to_int = dict(s = 1,b = 0)
    def convert(s):
        return to_int.get(s.decode("utf-8") , 0)
    
    data = data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1],
                                converters={1: convert})
    
    return data

def load_data_features(path_dataset):
    """load data features and creates an array with ids of the events"""
    data = data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, 
                                usecols=tuple(range(2,32)))
    
    ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[0])
    
    return data, ids



"""Data preprocessing
process data is the main function, it uses:
-create_subsets, creates subsets based on jet number
-remove_zero_variance, removes columnswithout variance
-normalize_data, normalizes the data based on the mean and on the std
-replace_missing, replaces -999 values by the median of the respective column
-correlation_filter, removes columns which correlate for less than treshold with the output"""


def process_data(X_train, X_test, y_train, y_test, ids_train, ids_test):
    """
    Processes the test and training data by:
    -splitting data with respect to jet number, creating three groups
    -removing zero variance in each subgroup
    -removing columns which are lowly correlated to y
    -normalizing the data with mean and standard devation 
    -replacing -999 by median value of column
    input:
        -X_train = ndarray of training dataset
        -X_test = ndarray of test dataset
        -y_train = 1darray of training goal-output
        -y_test = 1darray of test goal-output
        -ids_train = 1darray of id numbers of events in training data
        -ids_test = 1darray of id numbers of events in test data
    output:
        -train_subsets = list of training subsets with changes applied
        -test_subsets = list of test subsets with changes applied
        -y_train = list of training goal-output subsets
        -y_test = list of test goal-output subsets
        -ids_train = list of training id numbers subsets
        -ids_test = list of test id numbers subsets
    """
      
    train_subsets, y_train, ids_train = create_subsets(X_train, y_train, ids_train)
    test_subsets, y_test, ids_test = create_subsets(X_test, y_test, ids_test)
    
    for i in range(3):
        # change training sets
        train_subsets[i], mask = remove_zero_variance(train_subsets[i])
        print("For subgroup",i,"The following columns were removed due to zero variance:",[i for i, x in enumerate(mask) if x])
        
        train_subsets[i], mean, sigma = normalize_data(train_subsets[i], mean = None, sigma = None) 
        train_subsets[i], median = replace_missing(train_subsets[i], median = None) 
        train_subsets[i], quality = correlation_filter(train_subsets[i], y_train[i], threshold = 0.01)
        print("For subgroup",i,"The following columns were kept after low correlation:",quality)
        print("Final shape of subset",i,"is:",train_subsets[i].shape)
        
        #change test sets accordingly to training sets
        test_subsets[i], _ = remove_zero_variance(test_subsets[i], mask)
        test_subsets[i], _, _ =  normalize_data(test_subsets[i], mean, sigma)
        test_subsets[i], _ = replace_missing(test_subsets[i], median)
        test_subsets[i] = test_subsets[i][:, quality]
        
    return train_subsets, test_subsets, y_train, y_test, ids_train, ids_test
        
def create_subsets(data, y, ids):
    """Creates four subsets based on the number of jets,
    which is 0, 1 and 2 or 3. 2 and 3 are put in one group,
    since they keep same features and have similar correlation patterns
    input:
        data = training/test data
        y = training/test goal-output
        ids = training/test id numbers corresponding to data and y
    output:
        data_subsets = list containing the 3 subdata sets
        y_subsets = list containing the 3 subgoal-output sets
        ids_subsets = list conatining the 3 subids sets
    """
    data_subsets = []
    y_subsets = []
    ids_subsets = []
    for i in range(3):
        if i ==2:
            mask = data[:,22] >= i
        else:
            mask = data[:,22] == i
        data_subsets.append(data[mask])
        if y is not None:
            y_subsets.append(y[mask])
            
        ids_subsets.append(ids[mask])
        
    return data_subsets, y_subsets, ids_subsets


def remove_zero_variance(data, mask = None):
    """removes zero variance columns based on the subset
    input:
        -data =  training/test data
        -mask = None or boolean with true values for columns which should be removed
    output:
        -data with zero variance columns removed
        -mask = columns with zero columns"""
        
    if mask is None:
        variance = np.var(data, axis = 0)
        mask = variance == 0
    return data[:, ~mask[:]], mask


def normalize_data(data, mean = None, sigma = None):
    """Standardizes the data
    input:
        -data = training/test data
        -mean = None or mean calculated on training data
        -sigma = None or std calculated on training data
    outpu:
        -ouptut = normalized data
        -mean = array containing mean of each column
        -sigma = array containing std of each column"""
        
    if mean is None:
        mean = np.nanmean(data[data != -999], axis = 0)
    
    if sigma is None:
        sigma = np.nanstd(data[data != -999], axis = 0)
    
    output = (data - mean)/sigma
    
    return output, mean, sigma


def replace_missing(data, median = None):        
    """replaces -999 by median value
    input:
        -data = training/test data
        -median = None or list of median values for each column in training data
    output:
        -data = training/test data with -999 replaced by median values
        -median = list of median value per column"""
    
    if median is None:
        median =[]
        for j in range(data.shape[1]):
            mask = data[:,j] != -999
            replace = np.median(data[mask,j])
            data[~mask,j] = replace
            median.append(replace)
    else:
        for j in range(data.shape[1]):
            mask = data[:,j] != -999
            data[~mask,j] = median[j]

    return data, median


def correlation_filter(X, y, threshold = 0.01):
    """Removes features which are correlated with y with less than threshold
    input:
        X = training data
        y = corresponding goal-output of training data
        threshold = minimum correlation value for a column to be kept
    output:
        trimmed training data and columns which are kept"""
    abs_corr = np.zeros(X.shape[1])
    for index, x in enumerate(X.T):
        abs_corr[index] = np.abs(np.corrcoef(y,x.T)[0,1])
        
    quality = np.where(abs_corr > threshold)
    
    return X[:,quality[0]], quality[0]


"""Finalizing result, predicitons need to be put back together in the right order.
This is done by stitch_solution"""


def stitch_solution(X_test, y_result, ids_test_group, ids_test):
    """
    Puts found y values back in right order for the complete data matrix,
    since it was split in four groups.
    input:
        X_test =  original, preprocessed test data
        y_result =  output of created model,
            list of three vectors containing predictions for group 1,2 and 3
        ids_test_group = list of three arrays of ids. Each array contains the ids of one subset
        ids_test = array of all the ids
    output:
        y_final = list of the found y_values in the right order
    """
    y_final=[]
    for i in range(X_test.shape[0]):
        if ids_test[i] in ids_test_group[0]:
            index = np.where(ids_test_group[0] == ids_test[i])
            y_final.append(y_result[0][index])
            
        elif ids_test[i] in ids_test_group[1]:
            index = np.where(ids_test_group[1] == ids_test[i])
            y_final.append(y_result[1][index])
            
        elif ids_test[i] in ids_test_group[2]:
            index = np.where(ids_test_group[2] == ids_test[i])
            y_final.append(y_result[2][index])
            
    return y_final
