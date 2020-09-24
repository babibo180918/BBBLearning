import numpy as np
from ..utils.activation import *

def propagate(w, b, lambd, X, Y, cost_function):
    m = X.shape[1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)
    if cost_function == "cross_entropy":
        cost = -1/m*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A), axis = 1, keepdims = True) + lambd/(2*m)*np.sum(w, axis=1, keepdims=True)
    dw = 1/m*np.dot((A-Y), X.T) - lambd/m*w
    db = 1/m*np.sum(A-Y, axis=1, keepdims = True)
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
