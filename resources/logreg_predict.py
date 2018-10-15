import pandas as pd
import numpy as np
import argparse, glob, sys
from log_reg import preprocess

def sigmoid(z):
    return 1/(1+np.exp(-z))


def prediction(path1, path2, output_column,dataset):
    pre = preprocess()
    data = pre.preprocess(path1, output_column,dataset)

    weights = pd.read_csv(path2)
    weights = weights.drop(weights.columns[0], axis=1)

    predictions = {}
    results = pd.DataFrame(index=data.index)
    for weight in weights.columns:
        predictions[weight] = pd.DataFrame(sigmoid(np.dot(data,weights[weight])))
        results = pd.concat([results, predictions[weight]], axis=1)

    results.columns = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

    final_results = pd.DataFrame(results.idxmax(axis=1))
    final_results.columns = ['New students houses']
    final_results.to_csv('houses.csv')


def main():
    "Entrypoint for training."

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--files',
        help='Files top analyse globstring',
        default='*.csv')

    parser.add_argument(
        '--weights',
        help='Files top analyse globstring',
        default='*.csv')

    args = parser.parse_args()

    files = glob.glob(args.files)
    weights = glob.glob(args.weights)

    path1 = files[0]
    path2 = weights[0]


    prediction(path1=path1,
        path2=path2,
        output_column='Hogwarts House',
        dataset='test')

main()
