import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

# Import helper functions related to dataset processing
from _datasets.utils import *


def NMSE(x, y):
    # Normalize both vectors before calculating the error
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)

    # Compute the mean squared difference between the vectors
    return keras.ops.mean(keras.ops.square(x - y), axis=-1)


def get_first_item(d):
    # Return the first item from the dictionary
    return d[next(iter(d.keys()))]