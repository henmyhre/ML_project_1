# -*- coding: utf-8 -*-
"""
Functions used to find optimal hyperparameters


TODO: loss, precision, recall variables are not used

"""

import numpy as np
from Augmenting_data import *
from implementations import *

def find_parameters(X_train, y_train, degrees, lambdas, methods):
    """Finding best parameters and losses per set

    input:
        X_train = ndarray of training data
        y_train =  1darray of goal output corresponding to the training data
        lamdas = list of 1darray of floats to use as lambdas
        degrees = list of 1darray of integers to use for polynomial expansion
        gamma = learning rate, only relevant if penalized logistic regresssion is used
    output:
        best_parameter_loss = list containg best combination of method, degree and lambda per subset
        losses_sets = list of arrays containing all losses for methods used.
        Array sizes are equal to the length of degrees, length of lambdas and length of methods.
        List length is equal to number of subsets, 3"""
    
    best_parameter_per_set = []
    losses_sets =[]
    
    for i in range(3):
        print("Testing for set",i)
        parameters, losses = hyper_optimizing(X_train[i], y_train[i],
                                methods, lambdas[i], degrees[i], 10**-10, 3000)
        
        best_parameter_per_set.append(parameters)
        losses_sets.append(losses)
    
    return best_parameter_per_set, losses_sets

def hyper_optimizing(X_train, y_train, methods = ["Ridge_regression"],
                     lambdas = [0.1], degrees = [1], gamma = 0.0000001,  max_iter = 3000):
    """Finds best lambda and degree to use on the given data, test possibilities are:
    -Ridge regression
    -Penalized Logistic regression
    input:
        X_train = ndarray of training data
        y_train =  1darray of goal output corresponding to the training data
        methods = list of methods to be tested for, either ridge 
                    regression or penalized logistic regression
        lamdas = 1darray of floats to use as lambdas
        degrees = 1darray of integers to use for polynomial expansion
        gamma = learning rate, only relevant if penalized logistic regresssion is used
        max_iter = integer of maxinum of iterations for gradient descent, 
            only relevant if penalized logistic regresssion is used
    output:
        best_parameters = 1d array containg best method, best degree for that method and best lambda
        all_losses = array with all the losses found while testing. Shape is:
                                        len(methods), len(degrees), len(lambdas)
        """
    
    # Check method names are correct
    if len([i for i in methods if i in ["Ridge_regression", "Penalized_logistic"]]) < len(methods):
        raise NameError("At least one method is wrong")
        
    all_losses = np.zeros((len(methods), len(degrees), len(lambdas)))
    best_parameters = []
    
    for degree_index, degree in enumerate(degrees):
        X_train_ex = add_features(X_train, degree = degree)

        for method_index, method in enumerate(methods):
            
            if method == "Ridge_regression":
                k_fold = 5
                all_losses = compute_all_losses(method, method_index,
                                                y_train, X_train_ex, lambdas, k_fold, degree_index, degree, all_losses)
                
            elif method == "Penalized_logistic":
                k_fold = 4
                all_losses = compute_all_losses(method, method_index, 
                                                y_train, X_train_ex, lambdas, k_fold, degree_index, degree, all_losses)
                
            
    min_loss = np.argmin(all_losses)
    min_loss = np.unravel_index(min_loss, (len(methods), len(degrees), len(lambdas)))
    best_parameters.append(methods[min_loss[0]])
    best_parameters.append(degrees[min_loss[1]])
    best_parameters.append(lambdas[min_loss[2]])  
        
    return best_parameters, all_losses

def compute_all_losses(method, method_index, y_train, X_train, lambdas, k_fold, degree_index, degree, all_losses):
    seed = 1
    k_indices = build_k_indices(y_train, k_fold, seed)

    print_method_status(method, degree)

    for index, lambda_ in enumerate(lambdas):
        losses_te = []
        for k in range(k_fold):
            if method == "Ridge_regression":
                loss_te = cross_validation_ridge(y_train, X_train, k_indices, k, lambda_)
            elif method == "Penalized_logistic":
                loss_te = cross_validation_logistic(y_train, X_train, k_indices, k, lambda_, gamma, max_iter)

            losses_te.append(loss_te)
        all_losses[method_index, degree_index, index] = np.mean(losses_te)
    
    # Show percentage of correct results for this degree
    min_lambda = lambdas[np.argmin(all_losses[method_index, degree_index,:])]
    print("Lowest loss is:", min(all_losses[method_index, degree_index,:]),"for lambda:", min_lambda)

    return all_losses



def print_method_status(method, degree):
    print("Start ", method, " test for degree", str(degree),"...")
    
def cross_validation_ridge(y, x, k_indices, k, lambda_):

    """return the loss of ridge regression.

    input:
        y = 1darray of goal output corresponding to the training data
        x = ndarray of training data
        k_indices = array containing k subsets of indices which will be used to split x and y in test and 
            training data
        k = integer, kth subset of k_indices will be used to assign test data
        lamda_ = float, parameter for the ridge regression
    output:
        loss_te = float, loss found for the lambda used. Calculated with sqrt(2*MSE)
        """
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k], :]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_train = y[tr_indice]
    x_train = x[tr_indice, :]
    
    w = ridge_regression(y_train, x_train, lambda_)
   
    # calculate the loss for train and test data:
    loss_te = np.sqrt(2*compute_mse(y_test, x_test, w))

    #show accuracy and Fscore
    y_new = x_test @ w
    F_score, accuracy = quantify_result(y_new, y_test)
    print("Accuracy is:",accuracy, "Fscore is:",F_score)
    
    return loss_te

    
def cross_validation_logistic(y, x, k_indices, k,lambda_, gamma,max_iter):
    """return the loss of ridge regression.

    input:
        y = 1darray of goal output corresponding to the training data
        x = ndarray of training data
        k_indices = array containing k subsets of indices which will be used to split x and y in test and 
            training data
        k = integer, kth subset of k_indices will be used to assign test data
        lamda_ = float, parameter for the ridge regression
    output:
        loss_te = float, loss found for the lambda used. Calculated with sqrt(2*MSE)
        w = 1darray with the model with the lowest loss
    """
    # split according to k_indices
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k], :]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_train = y[tr_indice]
    x_train = x[tr_indice, :]
    
    loss, w = logistic_regression_penalized_gradient_descent(y_train, x_train, lambda_, gamma,max_iter)
        
    # calculate the loss for train and test data:
    loss_te = calculate_loss(y_test, x_test, w)
    
    # Calculate F_score
    y_new = x_test @ w
    F_score, accuracy = quantify_result(y_new, y_test)
    print("Accuracy is:",accuracy, "Fscore is:",F_score)
    
    return loss_te


def quantify_result(y_found, y_real):
    """Quantifies the result of y_found by comparing it to the real output, y_real
    
    input:
        y_found =  1darray containing the raw output of the model used
        y_real = 1darray with the goal output. Signal = 1, background = 0
    output:
        F_score, accuracy = numbers calculated by comparing y_real to binarized y_found"""
    y_found[y_found<0.5]=0
    y_found[y_found>=0.5]=1
    summ = y_found + y_real
    TP = np.sum(summ == 2)
    TN = np.sum(summ == 0)
    diff = y_found - y_real
    FP = np.sum(diff == 1)
    FN = np.sum(diff == -1)
    accuracy = (TP+TN)/(TP +TN +FP + FN)
    F_score = TP/(TP + 0.5 * (FP +FN))
    prec = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("Precision is:",prec,"Recall is:",recall)
    return F_score, accuracy


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


