import numpy as np


# Compute Cost
def compute_cost(AL, Y):
    m = AL.shape[1]

    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost
