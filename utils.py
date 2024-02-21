import os

import skops.io as sio

import custom_types

def save_model(filepath: str, model: custom_types.Scikit_Classifier) -> None:
    
    if not filepath.endswith('.skops'):
        raise ValueError("File name was not specified correctly: use .skops file extension")
    
    dir, _ = os.path.split(filepath)

    if not dir == '' and not os.path.exists(dir):
        os.mkdir(dir)

    sio.dump(model, filepath)