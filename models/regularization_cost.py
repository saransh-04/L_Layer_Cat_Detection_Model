from models.cost_function import *

import numpy as np


# Compute Cost with Regularization
def compute_cost_regularization(AL, Y, parameters, lambd):
    L = len(parameters) // 2
    regularization_value = 0
    m = AL.shape[1]

    cost_without_regularization = compute_cost(AL, Y)

    for l in range(1, L):

        regularization_value = regularization_value + np.sum(np.square(parameters["W" + str(l)]))

    regularization_value = (1 / m) * (lambd / 2) * regularization_value

    cost = cost_without_regularization + regularization_value

    return cost
