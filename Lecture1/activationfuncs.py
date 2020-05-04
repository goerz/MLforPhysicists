"""Collection of activation functions."""
import numpy as np


def sigmoid(z):
    """Sigmoid activation function.

    Maps (-∞,+∞) to (0, 1), centered around zero (sigmoid(0) = 0.5).
    """
    return 1 / (1 + np.exp(-z))


def jump(z):
    """Jump activation function.

    Maps negative negative numbers to zero, positive numbers to one
    """
    return np.array(z > 0, dtype='float')


def linear(z):
    """Linear activation function."""
    return z


def reLU(z):
    """Rectified Linear Unit function.

    Sets negative values to zero.
    """
    return (z > 0) * z
