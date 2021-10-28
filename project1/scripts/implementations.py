"""
File with all the basic implementations of the machine learning algorithms
-Linear gradient descent
-Linear stochastic gradient descent
-Leastsquared
-ridge regression
-logistic regression
-regularized logistic regression
"""

import numpy as np

"""Gradient descent
Uses the functions:
-compute_loss
-compute_gradient"""

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

def compute_loss(y, tx, w, method = "mse"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = len(y)
    e = y - np.dot(tx, w)
    
    if method == "mse":
        loss =(1/(2*N)) * np.dot(e.T,e)
        
    elif method == "mae":
        loss = np.mean(np.abs(e))
        
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    N = len(y)
    e = y - np.dot(tx, w)
    return -(1/N) * np.dot(tx.T,e)


"""Stochastic gradient descent
Uses the functions:
-compute_loss
-compute_gradient
-batch_iter"""

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(y, tx, w)
            # ***************************************************
            # update w by gradient
            w = w - gamma * gradient   
            
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    # ***************************************************
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
            
"""Least squares"""

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w


"""Ridge regression"""

def ridge_regression(y, tx, lambda_):
    """ridge regression"""
    if len(tx.shape) > 1:
        w = np.linalg.solve(tx.T @ tx + (2*tx.shape[0]*lambda_)*np.identity(tx.shape[1]), tx.T @ y)
    else:
        w = 1/(tx.T @ tx + lambda_) * tx.T @ y                        

    return w

"""Logistic regression
Uses the functions: sigmoid, calculate_loss, calculate_gradient"""

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    loss = calculate_loss(y, tx, w)
    # ***************************************************
    gradient = calculate_gradient(y, tx, w)
    # ***************************************************
    w = w - gradient*gamma
    # ***************************************************
    return loss, w


def sigmoid(t):
    """applies the sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """computes the loss: negative log likelihood."""
    inter_y = y.reshape(len(y),1)
    z = tx @ w
    a = np.sum(np.log(1 + np.exp(z)))
    b = inter_y.T @ z
    loss = a - b
    return np.squeeze(loss)

def calculate_gradient(y, tx, w):
    """computes the gradient of loss."""
    inter_y = y.reshape(len(y),1)
    gradient = tx.T @ (sigmoid(tx @ w) - inter_y)
    return gradient

"""Regularized lostic regression
Uses the functions: sigmoid, calculate_loss, calculate_gradient"""


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w -= gradient*gamma
    
    return loss, w

def logistic_regression_penalized_gradient_descent(y, x, lambda_, gamma,max_iter):
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return loss, w