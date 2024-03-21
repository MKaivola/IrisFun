import argparse



import utils

parser = argparse.ArgumentParser()

parser.add_argument("filename_new_data",
        type=str, 
        help="Path to a csv file containing the unlabeled data")
parser.add_argument("model_file",
        type=str,
        help="Path to the file where the learned model is stored")

if __name__ == '__main__':
    args = parser.parse_args()

    learned_model = utils.load_model(args.model_file)

    