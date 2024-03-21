import os

import pandas as pd
import skops.io as sio

import custom_types

def load_model(filepath: str) -> custom_types.Scikit_Classifier:
    """
    Load learned classifier from a given file. 
    Checks for unknown types before loading.

    Arguments
    ---------
    filepath 
        A string specifying the filepath to the model

    """

    if not filepath.endswith('.skops'):
        raise ValueError("File name was not specified correctly: use .skops file extension")

    if not os.path.exists(filepath):
        raise ValueError("The file does not exist")
    
    unknown_types = sio.get_untrusted_types(file = filepath)

    if unknown_types:
        print(f"The following unknown types were detected: {unknown_types}")

        answer = None

        while answer not in ('Y','n'):
            answer = input("Are these types trusted? ([Y]es/[n]o)")

        if answer == 'n':
            raise KeyboardInterrupt("The model file was not loaded, program interrupted")
        
    sio.load(filepath, trusted = unknown_types)