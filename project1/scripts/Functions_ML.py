# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:36:55 2021

@author: jurri
File with all the functions of ML
"""
import csv
import numpy as np

"""Data loading"""
def load_results(path_dataset):
    """load data features."""
    to_int = dict(s=1,b=-1)
    def convert(s):
        return to_int.get(s.decode("utf-8") , 0)
    
    data = data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1],
                                converters={1: convert})
    
    
    
    return data

def load_data_features(path_dataset):
    """load data features."""
    data = data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, 
                                usecols=tuple(range(2,32)))
    
    ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[0])
    
    return data, ids


""" Linear regression functions """
def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    N = len(y)
    e = y - np.dot(tx, w)
    
    return (1/(2*N)) * np.dot(e.T,e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    N = len(y)
    e = y - np.dot(tx, w)
    return -(1/N) * np.dot(tx.T,e)      


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # ***************************************************
        # update w by gradient
        w = w - gamma * gradient
        # ***************************************************
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(y, tx, w)
            # update w by gradient
            w = w - gamma * gradient   
            
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return losses, ws

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    Extended = np.empty((x.shape[0],0))
    for j in range(0, degree+1):
        for i in range(x.shape[1]):
            Extended = np.c_[Extended, x[:,i]**j]
    return Extended

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    if len(tx.shape) > 1:
        w = np.linalg.solve(tx.T @ tx + (2*tx.shape[0]*lambda_)*np.identity(tx.shape[1]), tx.T @ y)
    else:
        w = 1/(tx.T @ tx + lambda_) * tx.T @ y                        
    # ***************************************************
    return w

def sigmoid(t):
    """apply the sigmoid function on t."""
    sig = 1/(1+np.exp(-t))
    return sig

def calculate_loss_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(np.dot(tx, w))
    loss = np.squeeze(-np.dot(y.T, np.log(pred)) + np.dot((1-y).T, np.log(1-pred)))
    return loss

def calculate_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    gradient = np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)
    return gradient

def learning_by_gradient_descent_log(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = calculate_gradient_log(y, tx, w)
    loss = calculate_loss_log(y, tx, w)
    w = w - gamma * grad
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    threshold = 1e-8
    for iter in range(max_iters):
      # get loss and update w.
      loss, w = learning_by_gradient_descent_log(y, tx, w, gamma)
      # log info
      if iter % 100 == 0:
          print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
      # converge criterion
      losses.append(loss)
      if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
          break

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss_log(y, tx, w) + lambda_*np.squeeze(w.T @ w)
    grad = calculate_gradient_log(y, tx, w) + 2*lambda_*w
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    threshold = 1e-8
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_train = y[tr_indice]
    x_train = x[tr_indice]

    # ***************************************************
    # form data with polynomial degree
    if degree > 0:
        x_train_ex = build_poly(x_train, degree)
        x_test_ex =  build_poly(x_test, degree) 
    else:
        x_train_ex, x_test_ex = x_train, x_test
    # ***************************************************
    # ridge regression: 
    w = ridge_regression(y_train, x_train_ex, lambda_)
    # ***************************************************
    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2*compute_loss(y_train, x_train_ex, w))
    loss_te = np.sqrt(2*compute_loss(y_test, x_test_ex, w))
    
    
    return loss_tr, loss_te

#%%
#path = 'C:\Users\jurri\OneDrive\Documenten\University\Exchange\ML\train.csv\'
   
X, _ = load_data_features("train.csv")
y = load_results("train.csv")
#%% Find which lambda to use and which degree is best

def find_best_lamda(x_train, y_train, k_fold = 10, degree = 2, seed = 1):
    lambdas = np.logspace(-4, 0, 30)
    degrees = range(1, 10)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_te = np.empty((len(lambdas), len(degrees)))
    # ***************************************************
    for index_degree, degree in enumerate(degrees):
            x_train_ex = build_poly(x_train, degree)
            #find best lambda:
            for lambda_index, lambda_ in enumerate(lambdas):
                losses_te = []
                for k in range(k_fold):
                    _, loss_te = cross_validation(y_train, x_train_ex, k_indices, k, lambda_, degree)
                    losses_te.append(loss_te)
                rmse_te[lambda_index,index_degree] = np.mean(losses_te)
        
    best_index = np.unravel_index(rmse_te.argmin(), rmse_te.shape)
    print(best_index)    
    return lambdas[best_index[0]], degrees[best_index[1]]

lambda_ = find_best_lamda(X,y)
w = ridge_regression(y, X, lambda_)



#%% Create prediction
Xtest, ids = load_data_features("test.csv")

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
y_pred = predict_labels(w, Xtest)

name = "first_try.csv"

create_csv_submission(ids, y_pred, name)
















