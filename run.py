from models.load_data import *
from models.model import *
from models.model_regularization import *
from models.plot_file import *
from models.predict import *

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# View Image
index = 18
plt.imshow(train_x_orig[index])
print("Y for this image: " + str(train_y[0, index]) + ". That is class - " +
      classes[train_y[0, index]].decode('utf-8') + "image.")
plt.show()

# Exploring our dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))
print("\n\n")

# Reshaping the training examples.
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardizing the data to keep the values in between 0 and 1
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("\n\n")

# L Layer Model Excecution
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

# Cost without Regularization
print("Cost without Regularization")
parameters, costs = L_layer_model(train_x,
                                  train_y,
                                  layers_dims,
                                  learning_rate=0.01,
                                  num_iterations=3000,
                                  print_cost=True)

# Plot of cost vs iterations
plot_costs(costs, learning_rate=0.01)

# print("Training Predict Error")
pred_train = predict(train_x, train_y, parameters)

# print("Test Predict Error")
pred_test = predict(test_x, test_y, parameters)

# Cost with Regularization
print("Cost with Regularization")
parameters, costs = L_layer_model_regularization(train_x, train_y,
                                                 layers_dims,
                                                 learning_rate=0.01,
                                                 lambd=0.0075,
                                                 num_iterations=1000,
                                                 print_cost=True)

# Plot of cost vs iterations
plot_costs(costs, learning_rate=0.01)

print("Training Predict Error")
pred_train = predict(train_x, train_y, parameters)

print("Test Predict Error")
pred_test = predict(test_x, test_y, parameters)
