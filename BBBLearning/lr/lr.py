import os
from os.path import isfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .propagate import propagate
from .update import update
from ..utils.activation import *

def model(size, cost_function="cross_entropy", optimizer="GradientDescent", initializer="random"):
    model = {}
    model["size"] = size
    model["cost_function"] = cost_function
    model["optimizer"] = optimizer
    model["initializer"] = initializer
    # parameter initialization
    parameters = parameters_initialize(size, initializer)
    model["parameters"] = parameters
    
    return model

def parameters_initialize(size, initializer = "zero"):
    b = 0
    if initializer == "zero":
        w = np.zeros((1, size))
    elif initializer == "random":
        w = np.random.randn(1, size)
    elif initializer == "xavier":
        w = np.random.randn(1, size)*np.sqrt(1/size)
    elif initializer == "he":
        w = np.random.randn(1, size)*np.sqrt(2/size)

    parameters = {"w": w,
                  "b": b}
    return parameters

def train(model, X, Y, minibatch_size, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    size = model["size"]
    cost_function = model["cost_function"]
    optimizer = model["optimizer"]
    initializer = model["initializer"]
    parameters = model["parameters"]
    w = parameters['w']
    b = parameters['b']

    vgrads = {"vdw":0.0,
              "vdb":0.0}
    sgrads = {"sdw":0.0,
              "sdb":0.0}
    
    costs = []
    numOfMiniBatch = X.shape[1]//minibatch_size
    if numOfMiniBatch*minibatch_size < X.shape[1]:
        numOfMiniBatch = numOfMiniBatch + 1
    for epoch in range(num_epochs):
        epoch_cost = 0.
        for m in range(0,numOfMiniBatch):
            start = m*minibatch_size
            end = min((m+1)*minibatch_size, X.shape[1])
            # compute gradient and cost
            grads, minibatch_cost = propagate(w, b, lambd, X[:,start:end], Y[:,start:end], cost_function)

            vgrads["vdw"] = beta1*vgrads["vdw"] + (1-beta1)*grads["dw"]
            vgrads["vdb"] = beta1*vgrads["vdb"] + (1-beta1)*grads["db"]
            v_corrected = 1 - np.power(beta1, epoch)
            
            sgrads["sdw"] = beta2*sgrads["sdw"] + (1-beta2)*np.multiply(grads["dw"], grads["dw"])
            sgrads["sdb"] = beta2*sgrads["sdb"] + (1-beta2)*np.multiply(grads["db"], grads["db"])
            s_corrected = 1 - np.power(beta2, epoch)
            
            # update parameters
            w, b = update(w, b, grads, vgrads, sgrads, learning_rate, epsilon, v_corrected, s_corrected, optimizer)
            epoch_cost += minibatch_cost / minibatch_size
        # save costs
        if epoch % 10 == 0:
            costs.append(epoch_cost)
            if print_cost:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            parameters['w'] = w
            parameters['b'] = b
            model["parameters"] = parameters
            model['minibatch_size'] = minibatch_size
            model['num_epochs'] = num_epochs
            model['lambda'] = lambd
            model['learning_rate'] = learning_rate
            model['beta1'] = beta1
            model['beta2'] = beta2
            model['epsilon'] = epsilon
            with open(model_path, 'wb') as out_file:
                np.save(out_file, model)
                out_file.close()
                print("Model saved!")
    # plot costs
    plt.plot(np.squeeze(costs), color='red', marker='o')
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    params = {"w":w,
              "b":b}
    return params, costs

def train2(model, dataset, minibatch_size, num_epochs, lambd, learning_rate, beta1, beta2, epsilon, model_path, print_cost=False):
    size = model["size"]
    cost_function = model["cost_function"]
    optimizer = model["optimizer"]
    initializer = model["initializer"]
    parameters = model["parameters"]
    w = parameters['w']
    b = parameters['b']
    
    vgrads = {"vdw":0.0,
              "vdb":0.0}
    sgrads = {"sdw":0.0,
              "sdb":0.0}
    costs = []

    p = Path(dataset)
    file_sz = os.stat(dataset).st_size
    f = p.open('rb')
    for epoch in range(num_epochs):
        f.seek(0)
        epoch_cost = 0.
        while f.tell() < file_sz:
            batch = np.load(f, allow_pickle=True)
            X = batch[()]['X']
            Y = batch[()]['y']
            numOfMiniBatch = X.shape[1]//minibatch_size
            if numOfMiniBatch*minibatch_size < X.shape[1]:
                numOfMiniBatch = numOfMiniBatch + 1        
            for m in range(0,numOfMiniBatch):
                start = m*minibatch_size
                end = min((m+1)*minibatch_size, X.shape[1])
                # compute gradient and cost
                grads, minibatch_cost = propagate(w, b, lambd, X[:,start:end], Y[:,start:end], cost_function)

                vgrads["vdw"] = beta1*vgrads["vdw"] + (1-beta1)*grads["dw"]
                vgrads["vdb"] = beta1*vgrads["vdb"] + (1-beta1)*grads["db"]
                v_corrected = 1 - np.power(beta1, epoch)
                    
                sgrads["sdw"] = beta2*sgrads["sdw"] + (1-beta2)*np.multiply(grads["dw"], grads["dw"])
                sgrads["sdb"] = beta2*sgrads["sdb"] + (1-beta2)*np.multiply(grads["db"], grads["db"])
                s_corrected = 1 - np.power(beta2, epoch)
                    
                # update parameters
                w, b = update(w, b, grads, vgrads, sgrads, learning_rate, epsilon, v_corrected, s_corrected, optimizer)                
                epoch_cost += minibatch_cost / minibatch_size

        # save costs
        if epoch % 10 == 0:
            costs.append(epoch_cost)
            if print_cost:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            parameters['w'] = w
            parameters['b'] = b
            model["parameters"] = parameters
            model['minibatch_size'] = minibatch_size
            model['num_epochs'] = num_epochs
            model['lambda'] = lambd
            model['learning_rate'] = learning_rate
            model['beta1'] = beta1
            model['beta2'] = beta2
            model['epsilon'] = epsilon
            with open(model_path, 'wb') as out_file:
                np.save(out_file, model)
                out_file.close()
                print("Model saved!")

    plt.plot(np.squeeze(costs), color='red', marker='o')
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    f.close()
    params = {"w":w,
              "b":b}
    return params, costs

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = np.load(file, allow_pickle=True)
        return model[()]

def predict(params, X):
    w = params['w']
    b = params['b']
    # number of samples
    m = X.shape[1]
    # output init
    Y_pred = np.zeros((1, m))
    # predict
    z = np.dot(w, X) + b
    a = sigmoid(z)
    Y_pred = (a > 0.5).astype(int)
    return Y_pred
