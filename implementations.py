import numpy as np

def compute_mse(y, tx, w):
    """
    y:
    x:
    w:
    return the MSE loss
    """
    # ***************************************************
    e = y - tx@w
    return (1/2)*np.mean(e**2)        
    # ***************************************************

def least_squares(y, tx):
    """
    y: 
    x:
    w:    
    return the least squares weights and the associated loss
    """
    # ***************************************************
    A = tx.T@tx
    b = tx.T@y    
    w = np.linalg.solve(A, b)
    e = y - tx@w
    loss = compute_mse(y, tx, w)
    return w,  loss
    # ***************************************************

def compute_gradient(y, tx, w):
    """
    y:
    tx:
    w:    
    """
    grad = -np.mean(tx.T@(y - tx@w))
    return grad
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    y:
    tx:
    initial_w:
    max_iters:
    gamma:
    return the optimal weights using gradient descent and the associated loss
    """
    # ***************************************************
    w = initial_w
    for n in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma*grad
    loss = compute_mse(y, tx, w)
    return w, loss
    # ***************************************************
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    y:
    tx:
    initial_w:
    max_iters:
    gamma:
    return the optimal weights using stochatic gradient descent and the associated loss
    """
    # ***************************************************
    w = initial_w
    for n in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            grad = compute_gradienty_batch, tx_batch, w)
            w -= gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss
    # *************************************************** 
    
def ridge_regression(y, tx, lambda_):
    """
    y:
    tx:
    lambda_:
    return the optimal weights using ridge regression and the associated loss
    """
    # ***************************************************
    A = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    return np.linalg.solve(A, b)
    # ***************************************************
    
def sigmoid(t):
    """apply the sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return 1 / (1 + np.exp(-t))
    
def logistic_loss(y, tx, w):
    """
    y:
    x:
    w:
    return the logistic loss
    """
    # ***************************************************
    sig = sigmoid(-tx.dot(w))
    return - (y.T.dot(np.log(sig)) + (1 - y).T.dot(np.log(1 - sig)))       
    # ***************************************************
    
def logistic_gradient(y, tx, w):
    """
    y:
    x:
    w:
    return the gradient of the logistic function
    """
    # ***************************************************
    sig = sigmoid(-tx.dot(w))
    return tx.T@(sig - y)      
    # ***************************************************
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    y:
    tx:
    initial_w:
    max_iters:
    gamma:
    return the optimal weights using gradient descent with logistic loss and the associated loss
    """
    # ***************************************************
    w = initial_w
    for n in range(max_iters):
        grad = logistic_gradient(y, tx, w)
        w = w - gamma*grad
    loss = logistic_loss(y, tx, w)
    return loss, w   
    # ***************************************************

def reglogistic_loss(y, tx, w, lambda_):
    """
    y:
    x:
    w:
    return the logistic loss
    """
    # ***************************************************
    return logistic_loss(y, tx, w) + (lambda_/2) * np.squeeze(w.T@w)
    # ***************************************************
    
def reglogistic_gradient(y, tx, w, lambda_):
    """
    y:
    x:
    w:
    return the gradient of the logistic function
    """
    # ***************************************************
    return logistic_gradient(y, tx, w)  + lambda_ * np.squeeze(w.T@w)
    # ***************************************************    
    
def reglogistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    y:
    tx:
    initial_w:
    max_iters:
    gamma:
    return the optimal weights using stochatic gradient descent with logistic loss and the associated loss
    """
    # ***************************************************
    w = initial_w
    for n in range(max_iters):
        grad = reglogistic_gradient(y, tx, w, lambda_)
        w = w - gamma*grad
    loss = reglogistic_loss(y, tx, w, lambda_)
    return loss, w   
    # ***************************************************

    
