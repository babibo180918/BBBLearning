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

def train(model, X, Y, iteration, learning_rate, print_cost=False):
    size = model["size"]
    cost_function = model["cost_function"]
    optimizer = model["optimizer"]
    initializer = model["initializer"]

    costs = []
    # parameter initialization
    w, b = parameters_initialize(initializer, size)
    for i in range(iteration):
        # compute gradient and cost
        grads, cost = propagate(w, b, X, Y, cost_function)

        # update parameters
        w, b = update(w, b, learning_rate, optimizer)

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
