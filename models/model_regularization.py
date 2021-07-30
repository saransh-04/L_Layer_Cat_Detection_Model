from models.parameters_file import *
from models.forward_prop import *
from models.regularization_cost import *
from models.regularization_back_prop import *


# L Layer Model Regularization
def L_layer_model_regularization(X, Y, layers_dims, learning_rate, lambd, num_iterations, print_cost=False):

    np.random.seed(1)
    costs = []

    # Initialize the parameters
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(num_iterations):
        # Forward Propagation
        AL, caches = L_model_forward(X, parameters)

        # Compute Cost
        cost = compute_cost_regularization(AL, Y, parameters, lambd)
        costs.append(cost)

        # Back Propagation
        grads = L_model_backward_regularization(AL, Y, caches, lambd)

        # Update Parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs
