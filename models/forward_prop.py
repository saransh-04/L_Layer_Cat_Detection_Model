from models.activation_functions import *


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