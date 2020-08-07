import numpy as np

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def relu(z):
    a = np.where(z<0, 0, z)
    return a

def softmax(z):
    a = np.exp(z)
    a = a/np.sum(a, axis = 0, keepdims = True)
    return a
