import math
import numpy as np
import h5py

def load_test_dataset():
    test_dataset = h5py.File('test_dataset/dataset/onepiece_64.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return test_set_x_orig, test_set_y_orig, classes
