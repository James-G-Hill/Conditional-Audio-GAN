from sklearn.neighbors import NearestNeighbors

import argparse as ag
import importlib.machinery as im
import numpy as np
import types


def nn_distance(args):
    """ Calculates the nearest neighbors distance """

    # Get data
    dataTransformer = _loadModule(
        'TransformData.py',
        'TransformData.py'
    )
    realData = dataTransformer.transform_data(
        args.trainDir,
        args.samRate,
        args.fileCount
    )
    genData = dataTransformer.transform_data(
        args.queryDir,
        args.samRate,
        args.fileCount
    )

    # Create models and calculate
    nn = NearestNeighbors(
        n_neighbors=1,
        algorithm='ball_tree'
    ).fit(realData)
    distances, _ = nn.kneighbors(genData)

    # Return results
    distances = distances[:, 0]

    # Print the results
    print('Mean          : ' + str(np.mean(distances)))
    print('Standard Error: ' + str(np.std(distances)))

    return


def _loadModule(modName, modPath):
    """ Loads the module containing the relevant networks """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


if __name__ == '__main__':

    parser = ag.ArgumentParser()
    parser.add_argument(
        '-trainDir',
        type=str,
        help='The training set for the KNN model.'
    )
    parser.add_argument(
        '-queryDir',
        type=str,
        help='The query set for the KNN model.'
    )
    parser.add_argument(
        '-count',
        type=int,
        help='The amount of real data to test.'
    )
    args = parser.parse_args()

    nn_distance(args)
