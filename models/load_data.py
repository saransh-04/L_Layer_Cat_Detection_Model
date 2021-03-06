import h5py
import numpy as np


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
