from models.backward_prop import *

import numpy as np


# Linear Backward with Regularization
def linear_backward_regularization(dZ, cache, lambd):
    # lambd is the regularization parameter

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dA_prev, dW, db = linear_backward(dZ, cache)

    dW = dW + (lambd / m) * W

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# Linear Activation Backward with Regularization
def linear_activation_backward_regularization(dA, cache, activation, lambd):
    global dA_prev, dW, db

    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_regularization(dZ, linear_cache, lambd)

    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_regularization(dZ, linear_cache, lambd)

    return dA_prev, dW, db


# L Layer Backward Propagation
def L_model_backward_regularization(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward_regularization(dAL, current_cache, "sigmoid", lambd)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = \
            linear_activation_backward_regularization(grads["dA" + str(l + 1)], current_cache, "relu", lambd)

    return grads
