import numpy as np

def model(size, cost_function="cross_entropy", optimizer="GradientDecent", initializer="random"):
    model = {}
    model["size"] = size
    model["cost_function"] = cost_function
    model["optimizer"] = optimizer
    model["initializer"] = initializer
    return model

def parameters_initialize(initializer = "zero", size):
    b = 0
    if initializer == "zero":
        w = np.zeros((size, 1))
    else if initializer == "random":
        w = np.random.randn(size, 1)
    else if initializer == "xavier":
        w = np.random.randn(size, 1)*np.sqrt(1/size)
    else if initializer == "he":
        w = np.random.randn(size, 1)*np.sqrt(2/size)
    return w, b

def train(model, X, Y, iteration, learning_rate, beta1, beta2, print_cost=False):
    size = model["size"]
    cost_function = model["cost_function"]
    optimizer = model["optimizer"]
    initializer = model["initializer"]

    costs = []
    vgrads = {"vdw":0.0,
              "vdb":0.0}
    sgrads = {"sdw":0.0,
              "sdb":0.0}
    # parameter initialization
    w, b = parameters_initialize(initializer, size)
    for i in range(iteration):
        # compute gradient and cost
        grads, cost = propagate(w, b, X, Y, cost_function)

        vgrads["vdw"] = beta1*vgrads["vdw"] + (1-beta1)*grads["dw"]
        vgrads["vdb"] = beta1*vgrads["vdb"] + (1-beta1)*grads["db"]
        v_corrected = 1 - np.power(beta1, i)
        
        sgrads["sdw"] = beta2*vgrads["sdw"] + (1-beta2)*np.multiply(grads["dw"], grads["dw"])
        sgrads["sdb"] = beta2*vgrads["sdb"] + (1-beta2)*np.multiply(grads["db"], grads["db"])
        s_corrected = 1 - np.power(beta2, i)
        
        # update parameters
        w, b = update(w, b, optimizer, grads, vgrads, sgrads, learning_rate, epsilon, v_corrected, s_corrected)

        # save costs
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f", %(i, cost))
    params = {"w":w,
              "b":b}
    return params, costs

def predict(params, X):
    # number of samples
    m = X.shape[1]
    # output init
    Y_pred = np.zeros((1, m))
    # predict
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    Y_pred = (a > 0.5).astype(int)
    return Y_pred
