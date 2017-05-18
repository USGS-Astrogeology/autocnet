from collections import deque
import math
import warnings

import numpy as np
import pandas as pd


def distance_ratio(matches, ratio=0.8, single=False):
    """
    Compute and return a mask for a matches dataframe
    using Lowe's ratio test.  If keypoints have a single
    Lowe (2004) [Lowe2004]_

    Parameters
    ----------
    ratio : float
            the ratio between the first and second-best match distances
            for each keypoint to use as a bound for marking the first keypoint
            as "good". Default: 0.8

    single : bool
             If True, points with only a single entry are included (True)
             in the result mask, else False.

    Returns
    -------
    mask : pd.dataframe
           A Pandas DataFrame mask for the matches with those failing the
           ratio test set to False.
    """
    def func(group):
        res = [False] * len(group)
        if len(res) == 1:
            return [single]
        if group.iloc[0] < group.iloc[1] * ratio:
            res[0] = True
        return res

    mask_s = matches.groupby('source_idx')['distance'].transform(func).astype('bool')
    single = True
    mask_d = matches.groupby('destination_idx')['distance'].transform(func).astype('bool')
    mask = mask_s & mask_d

    return mask


def spatial_suppression(df, domain, min_radius=1.5, k=250, error_k=0.1):
    """
    Spatial suppression using disc based method.

    Attributes
    ----------
    df : dataframe
         Input dataframe used for suppressing

    mask : series
           pandas boolean series

    max_radius : float
                 Maximum allowable point radius

    min_radius : float
                 The smallest allowable radius size

    nvalid : int
             The number of valid points after suppression

    k : int
        The number of points to be saved

    error_k : float
              [0,1] the acceptable error in k

    domain : tuple
             The (x,y) extent of the input domain

    Returns
    -------
    mask : pd.Series
           Boolean suppression mask

    k : int
        The number of unsuppressed observations

    References
    ----------
    [Gauglitz2011]_

    """
    columns = df.columns
    for i in ['x', 'y', 'strength']:
        if i not in columns:
            raise ValueError('The dataframe is missing a {} column.'.format(i))
    df = df.sort_values(by=['strength'], ascending=False).copy()
    max_radius = max(domain)
    mask = pd.Series(False, index=df.index)

    process = True
    if k > len(df):
        warnings.warn('Only {} valid points, but {} points requested'.format(len(df), k))
        k = len(df)
        result = df.index
        process = False
    nsteps = max(domain) * 0.95
    search_space = np.linspace(min_radius, max_radius, nsteps)
    cell_sizes = search_space / math.sqrt(2)
    min_idx = 0
    max_idx = len(search_space) - 1

    prev_min = None
    prev_max = None

    while process:
        # Setup to store results
        result = []

        mid_idx = int((min_idx + max_idx) / 2)

        if min_idx == mid_idx or mid_idx == max_idx:
            warnings.warn('Unable to optimally solve.  Returning with {} points'.format(len(result)))
            process = False

        cell_size = cell_sizes[mid_idx]
        n_x_cells = int(domain[0] / cell_size)
        n_y_cells = int(domain[1] / cell_size)
        grid = np.zeros((n_x_cells, n_y_cells), dtype=np.bool)

        # Assign all points to bins
        x_edges = np.linspace(0, domain[0], n_x_cells)
        y_edges = np.linspace(0, domain[1], n_y_cells)
        xbins = np.digitize(df['x'], bins=x_edges)
        ybins = np.digitize(df['y'], bins=y_edges)

        # Convert bins to cells
        xbins -= 1
        ybins -= 1
        pts = []
        for i, (idx, p) in enumerate(df.iterrows()):
            x_center = xbins[i]
            y_center = ybins[i]
            cell = grid[y_center, x_center]

            if cell == False:
                result.append(idx)
                pts.append((p[['x', 'y']]))
                if len(result) > k + k * error_k:
                    # Too many points, break
                    min_idx = mid_idx
                    break

                y_min = y_center - int(round(cell_size, 0))
                if y_min < 0:
                    y_min = 0

                x_min = x_center - int(round(cell_size, 0))
                if x_min < 0:
                    x_min = 0

                y_max = y_center + int(round(cell_size, 0))
                if y_max > grid.shape[0]:
                    y_max = grid.shape[0]

                x_max = x_center + int(round(cell_size, 0))
                if x_max > grid.shape[1]:
                    x_max = grid.shape[1]

                # Cover the necessary cells
                grid[y_min: y_max,
                     x_min: x_max] = True

        #  Check break conditions
        if k - k * error_k <= len(result) <= k + k * error_k:
            process = False
        elif len(result) < k - k * error_k:
            # The radius is too large
            max_idx = mid_idx
            if max_idx == 0:
                warnings.warn('Unable to retrieve {} points. Consider reducing the amount of points you request(k)'
                              .format(k))
                process = False
            if min_idx == max_idx:
                process = False
    mask = pd.Series(False, df.index)
    mask.loc[list(result)] = True

    return mask, k


def self_neighbors(matches):
    """
    Returns a pandas data series intended to be used as a mask. Each row
    is True if it is not matched to a point in the same image (good) and
    False if it is (bad.)

    Parameters
    ----------
    matches : dataframe
              the matches dataframe stored along the edge of the graph
              containing matched points with columns containing:
              matched image name, query index, train index, and
              descriptor distance
    Returns
    -------
    : dataseries
      Intended to mask the matches dataframe. True means the row is not matched to a point in the same image
      and false the row is.
    """
    return matches.source_image != matches.destination_image


def mirroring_test(matches):
    """
    Compute and return a mask for the matches dataframe on each edge of the graph which
    will keep only entries in which there is both a source -> destination match and a destination ->
    source match.

    Parameters
    ----------
    matches : dataframe
              the matches dataframe stored along the edge of the graph
              containing matched points with columns containing:
              matched image name, query index, train index, and
              descriptor distance

    Returns
    -------
    duplicates : dataseries
                 Intended to mask the matches dataframe. Rows are True if the associated keypoint passes
                 the mirroring test and false otherwise. That is, if 1->2, 2->1, both rows will be True,
                 otherwise, they will be false. Keypoints with only one match will be False. Removes
                 duplicate rows.
    """
    duplicate_mask = matches.duplicated(subset=['source_idx', 'destination_idx', 'distance'], keep='last')
    return duplicate_mask
