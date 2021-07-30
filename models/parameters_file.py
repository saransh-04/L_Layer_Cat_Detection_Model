import numpy as np

np.random.seed(1)


# Initializing Parameters
def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters


# HE initialization
def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):

        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        # * np.sqrt(2 / layer_dims[l - 1]) or * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


# Update Parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
