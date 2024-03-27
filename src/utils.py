import pickle
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


### saving and loading made-easy
def save(pickle_file, array):
    """
    Pickle array (in general any formattable object)
    args::
        pickle_file: str, path to save the file
        array: object to be pickled
    ret::
        None
    """
    with open(pickle_file, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(pickle_file):
    """
    Loading pickled array
    args::
        pickle_file: str, path to load the file
    ret::
        b: object loaded from pickle file
    """
    with open(pickle_file, 'rb') as handle:
        b = pickle.load(handle)
    return b