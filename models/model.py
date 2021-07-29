import numpy as np
import matplotlib.pyplot as plt
import h5py


# Load Data Function
def load_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("data/test_catvnoncat.h5", 'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(train_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Initializing Parameters
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# Sigmoid Function
def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z

    assert (A.shape == Z.shape)

    return A, cache


# ReLU Function
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    assert (A.shape == Z.shape)

    return A, cache


# Linear Forward:
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


# Linear Activation Forward
def linear_activation_forward(A_prev, W, b, activation):
    global linear_cache, activation_cache
    A = []

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)
    # linear_cache = (A, W, b)
    # activation_cache = Z

    return A, cache


# Forward Prop for L layers
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

        assert (A.shape == (parameters["W" + str(l)].shape[0], A_prev.shape[1]))

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    # Finally there will be only output for every example.

    return AL, caches


# Compute Cost
def compute_cost(AL, Y):
    m = AL.shape[1]

    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


# Sigmoid Backward Function
def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = np.multiply(dA, np.multiply(s, (1 - s)))

    assert (dZ.shape == Z.shape)

    return dZ


# ReLU Backward Function
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


# Linear Backward
def linear_backward(dZ, cache):
    # Linear Cache

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# Linear Activation Baackward
def linear_activation_backward(dA, cache, activation):
    global dA_prev, dW, db

    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# L Layer Backward Propagation
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")

    return grads


# Update Parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# L Layer Model

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    # (≈ 1 line of code)
    # parameters = ...
    # YOUR CODE STARTS HERE

    parameters = initialize_parameters(layers_dims)

    # YOUR CODE ENDS HERE

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


# Plotting the Cost
def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# Predict Function
def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == Y) / m)))

    return p
