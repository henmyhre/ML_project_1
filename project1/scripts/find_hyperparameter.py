# -*- coding: utf-8 -*-
"""
Functions used to find optimal hyperparameters
"""

import numpy as np
from Augmenting_data import *
from implementations import *

def find_parameters(X_train, y_train, degrees, lambdas, methods):
    """Finding best parameters and losses per set
    input:
        X_train = ndarray of training data
        y_train =  1darray of goal output corresponding to the training data
        lamdas = 1darray of floats to use as lambdas
        degrees = 1darray of integers to use for polynomial expansion
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
                                methods, lambdas, degrees, 10**-10, 3000)
        
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
    best_parameters = np.zeros((3,1))
    
    for degree_index, degree in enumerate(degrees):
        X_train_ex = add_features(X_train, degree = degree)
    
        for method_index, method in enumerate(methods):
            
            if method == "Ridge_regression":
                seed = 1
                k_fold = 5
                k_indices = build_k_indices(y_train, k_fold, seed)
                print("Start ridge regression test for degree", str(degree),"...")
                for index, lambda_ in enumerate(lambdas):
                    losses_te = []
                    for k in range(k_fold):
                        loss_te = cross_validation_ridge(y_train, X_train_ex, k_indices, k, lambda_)
                        losses_te.append(loss_te)
                    all_losses[method_index, degree_index, index] = np.mean(losses_te)
                 
                # Show percantage of correct results for this degree
                min_lambda = lambdas[np.argmin(all_losses[method_index, degree_index,:])]
                print("Lowest loss is for lambda:", min_lambda, "is:", min(all_losses[method_index, degree_index,:]))
                
                
            elif method == "Penalized_logistic":
                #less k-fold for reason of speed
                seed = 1
                k_fold = 4
                k_indices = build_k_indices(y_train, k_fold, seed)
                print("Start penalized_logistic test...")
                for index, lambda_ in enumerate(lambdas):
                    losses_te = []
                    for k in range(k_fold):
                        loss_te, _ = cross_validation_logistic(y_train, X_train_ex, k_indices,
                                                    k, lambda_, gamma, max_iter)
                        losses_te.append(loss_te)
                    all_losses[method_index, degree_index, index] = np.mean(losses_te) 
                
                # Show percantage of correct results for this degree
                min_lambda = lambdas[np.argmin(all_losses[method_index, degree_index,:])]
                print("Lowest loss is:", min(all_losses[method_index, degree_index,:]),"for lambda:", min_lambda)
                
            
    min_loss = np.argmin(all_losses)
    min_loss = np.unravel_index(min_loss, (len(methods), len(degrees), len(lambdas)))
    best_parameters[0] = methods[min_loss[0]]
    best_parameters[1] = degrees[min_loss[1]]
    best_parameters[2] = lambdas[min_loss[2]]  
        
    return best_parameters, all_losses


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
    # Calculate F_score, seems more reliable to compare different degrees
    #y_new = x_test @ w
    #precision, recall, F_score, accuracy = quantify_result(y_new, y_test)
    
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
    
    w = np.zeros((x.shape[1], 1))
    threshold = 1e-8
    losses = []
    # start the logistic regression
    iter = 0
    while iter <max_iter:
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y_train, x_train, w, gamma, lambda_)
        # log info
        if iter % 999 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        # check loss actually decreases, if not decrease gamma
        if iter > 0:
            if loss > losses[-1]:
                gamma = gamma/2
            if np.isinf(loss):
                iter = 0
                w = np.zeros((x.shape[1], 1))
                gamma = gamma/10
                
        iter +=1
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    # calculate the loss for train and test data:
    loss_te = calculate_loss(y_test, x_test, w)
    
    # Calculate F_score, seems more reliable to compare different degrees
    y_new = x_test @ w
    precision, recall, F_score, accuracy = quantify_result(y_new, y_test)
    print(accuracy, F_score)
    
    return loss_te, w

def quantify_result(y_found, y_real):
    """Quantifies the result of y_found by comparing it to the real output, y_real
    input:
        y_found =  1darray containing the raw output of the model used
        y_real = 1darray with the goal output. Signal = 1, background = 0
    output:
        precision, recall, F_score, accuracy = numbers calculated by comparing y_real to binarized y_found"""
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
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    return precision, recall, F_score, accuracy


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


