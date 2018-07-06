from collections import deque
import math
import warnings

import numpy as np
import pandas as pd


def distance_ratio(edge, matches, ratio=0.8, single=False):
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


def spatial_suppression(df, bounds, xkey='x', ykey='y', k=60, error_k=0.05, nsteps=250):
    """
    Apply the spatial suppression algorithm over an arbitrary domain for all of the spatial
    data in the provided data frame.

    Parameters
    ----------

    df : object
         Pandas data frame with coordinates

    bounds : list
             In the form xmin, ymin, xmax, ymax

    xkey : str
           The column name for the x coordinates

    ykey : str
           The column name for the y coordinates

    k : int
        The desired number of points after suppression

    error_k : float
              The percentage of allowable error in the domain [0,1]

    nsteps : int
             The granularity of the search. This controls the number of
             buckets in the x and y dimension. More granular search adds processing
             time, but can result in a more accurate solution.

    Returns
    -------
    mask : nd.array
           A boolean mask of the valid points

    len(result) : int
                  The numer of valud points
    """
    # Compute the bounding area inside of which the suppression will be applied
    minx = min(bounds[0], bounds[2])
    maxx = max(bounds[0], bounds[2])
    miny = min(bounds[1], bounds[3])
    maxy = max(bounds[1], bounds[3])
    domain = (maxx-minx),(maxy-miny)

    min_radius = min(domain) / 20
    max_radius = max(domain)
    search_space = np.linspace(min_radius, max_radius, nsteps)
    cell_sizes = search_space / math.sqrt(2)
    min_idx = 0
    max_idx = len(search_space) - 1

    # Setup flags to watch for looping
    prev_min = None
    prev_max = None

    # Sort the dataframe (hard coded to ascending as lower strength (cost) is better)
    df = df.sort_values(by=['strength'], ascending=True).copy()
    df = df.reset_index(drop=True)
    mask = pd.Series(False, index=df.index)

    process = True
    while process:
        # Binary search
        mid_idx = int((min_idx + max_idx) / 2)
        if min_idx == mid_idx or mid_idx == max_idx:
            warnings.warn('Unable to optimally solve.')
            process = False
        else:
            # Setup to store results
            result = []

        # Get the current cell size and grid the domain
        cell_size = cell_sizes[mid_idx]
        n_x_cells = int(round(domain[0] / cell_size, 0)) - 1
        n_y_cells = int(round(domain[1] / cell_size, 0)) - 1

        if n_x_cells <= 0:
            n_x_cells = 1
        if n_y_cells <= 0:
            n_y_cells = 1

        grid = np.zeros((n_y_cells, n_x_cells), dtype=np.bool)
        # Assign all points to bins
        x_edges = np.linspace(minx, maxx, n_x_cells)
        y_edges = np.linspace(miny, maxy, n_y_cells)
        xbins = np.digitize(df[xkey], bins=x_edges)
        ybins = np.digitize(df[ykey], bins=y_edges)

        # Starting with the best point, start assigning points to grid cells
        for i, (idx, p) in enumerate(df.iterrows()):
            x_center = xbins[i] - 1
            y_center = ybins[i] - 1
            cell = grid[y_center, x_center]

            if cell == False:
                result.append(idx)
                # Set the cell to True
                grid[y_center, x_center] = True

            # If everything is already 'covered' break from the list
            if grid.all() == False:
                continue

        # Check to see if the algorithm is completed, or if the grid size needs to be larger or smaller
        if k - k * error_k <= len(result) <= k + k * error_k:
            # Success, in bounds
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

        elif len(result) > k + k * error_k:
            # Too many points, break
            min_idx = mid_idx

    mask.loc[list(result)] = True

    return mask, len(result)


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
