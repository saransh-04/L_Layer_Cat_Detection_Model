from models.parameters_file import *
from models.forward_prop import *
from models.cost_function import *
from models.backward_prop import *


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
