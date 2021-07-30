import numpy as np


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
