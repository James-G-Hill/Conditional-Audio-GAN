from sklearn.neighbors import NearestNeighbors
import argparse as ag
import numpy as np


def nn_distance(train, query):
    """ Calculates the nearest neighbors distance """

    # Create models and calculate
    nn = NearestNeighbors(
        n_neighbors=1,
        algorithm='ball_tree'
    ).fit(train)
    distances, _ = nn.kneighbors(query)

    # Return results
    distances = distances[:, 0]

    # Print the results
    print('Mean          : ' + str(np.mean(distances)))
    print('Standard Error: ' + str(np.std(distances)))

    return


if __name__ == '__main__':

    parser = ag.ArgumentParser()
    parser.add_argument(
        '-train',
        type=str,
        help='The training set for the KNN model.'
    )
    parser.add_argument(
        '-query',
        type=str,
        help='The query set for the KNN model.'
    )
    args = parser.parse_args()

    nn_distance(args.train, args.query)
