import numpy as np


def response(row, edge):
    """
    Suppression function that converts 'response' into 'strength'
    """
    return row['response']


def correlation(row, edge):
    """
    Suppression function that converts 'correlation' into 'strength'
    """
    return row['correlation']


def distance(row, edge):
    """
    Suppression function that converts 'distance' into 'strength'
    """
    return 1 / row['distance']


def error(row, edge):
    """
    Suppression function that takes the reprojection error
    in a fundamental matrix as the inverse strength.
    """
    key = row.name

    try:
        return 1 / edge.fundamental_matrix.error.loc[key]
    except:
        return np.NaN
