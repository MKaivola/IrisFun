import argparse

import pandas as pd

import utils_predict

parser = argparse.ArgumentParser()

parser.add_argument("filename_new_data",
        type=str, 
        help="Path to a csv file containing the unlabeled data")
parser.add_argument("model_file",
        type=str,
        help="Path to the file where the learned model is stored")
parser.add_argument("output",
        type=str,
        help="Path to the desired csv output file")

if __name__ == '__main__':
    args = parser.parse_args()

    unlabeled_data = pd.read_csv(args.filename_new_data)

    learned_model = utils_predict.load_model(args.model_file)

    utils_predict.validate_input(unlabeled_data, learned_model)

    X = utils_predict.preprocess_input(unlabeled_data)

    predicted_labels = learned_model.predict(X)

    labeled_data = unlabeled_data.assign(y_pred = predicted_labels)

    labeled_data.to_csv(args.output, sep = ',', index = False)