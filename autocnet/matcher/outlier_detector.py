from collections import deque
import math
import warnings

import cv2
import numpy as np
import pandas as pd

from autocnet.utils.observable import Observable


class DistanceRatio(Observable):

    """
    A stateful object to store ratio test results and provenance.

    Attributes
    ----------

    nvalid : int
             The number of valid entries in the mask

    mask : series
           Pandas boolean series indexed by the match id

    matches : dataframe
              The matches dataframe from an edge.  This dataframe
              must have 'source_idx' and 'distance' columns.

    single : bool
             If True, then single entries in the distance ratio
             mask are assumed to have passed the ratio test.  Else
             False.

    References
    ----------
    [Lowe2004]_

    """

    def __init__(self, matches):

        self._action_stack = deque(maxlen=10)
        self._current_action_stack = 0
        self._observers = set()
        self.matches = matches
        self.mask = None
        self.clean_keys = None
        self.single = None
        self.attrs = ['mask', 'ratio', 'clean_keys', 'single']

    @property
    def nvalid(self):
        return self.mask.sum()

    def compute(self, ratio=0.8, mask=None, mask_name=None, single=False):
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

        mask : series
               A pandas boolean series to initially mask the matches array

        mask_name : list or str
                    An arbitrary mask name for provenance tracking

        single : bool
                 If True, points with only a single entry are included (True)
                 in the result mask, else False.
        """
        def func(group):
            res = [False] * len(group)
            if len(res) == 1:
                return [single]
            if group.iloc[0] < group.iloc[1] * ratio:
                res[0] = True
            return res

        self.single = single
        if mask is not None:
            self.mask = mask.copy()
            new_mask = self.matches[mask].groupby('source_idx')['distance'].transform(func).astype('bool')
            self.mask[mask == True] = new_mask
        else:
            self.mask = self.matches.groupby('source_idx')['distance'].transform(func).astype('bool')

        state_package = {'ratio': ratio,
                         'mask': self.mask.copy(),
                         'clean_keys': mask_name,
                         'single': single
                         }

        self._action_stack.append(state_package)
        self._current_action_stack = len(self._action_stack) - 1


class SpatialSuppression(Observable):
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

    References
    ----------
    [Gauglitz2011]_

    """

    def __init__(self, df, domain, min_radius=2, k=250, error_k=0.1):
        columns = df.columns
        for i in ['x', 'y', 'strength']:
            if i not in columns:
                raise ValueError('The dataframe is missing a {} column.'.format(i))
        self._df = df.sort_values(by=['strength'], ascending=False).copy()
        self.max_radius = max(domain)
        self.min_radius = min_radius
        self.domain = domain
        self.mask = None
        self._k = k
        self._error_k = error_k

        self.attrs = ['mask', 'k', 'error_k']

        self._action_stack = deque(maxlen=10)
        self._current_action_stack = 0
        self._observers = set()

    @property
    def nvalid(self):
        try:
            return self.mask.sum()
        except:
            return None

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, v):
        self._k = v

    @property
    def error_k(self):
        return self._error_k

    @error_k.setter
    def error_k(self, v):
        self._error_k = v

    @property
    def df(self):
        return self._df

    def suppress(self):
        """
        Suppress subpixel registered points to that k +- k * error_k
        points, with good spatial distribution, remain
        """
        if self.k > len(self.df):
            warnings.warn('Only {} valid points, but {} points requested'.format(len(self.df), self.k))
            self.k = len(self.df)
        search_space = np.linspace(self.min_radius, self.max_radius / 16, 250)
        cell_sizes = (search_space / math.sqrt(2)).astype(np.int)
        min_idx = 0
        max_idx = len(search_space) - 1
        while True:
            mid_idx = int((min_idx + max_idx) / 2)
            r = search_space[mid_idx]

            cell_size = cell_sizes[mid_idx]
            n_x_cells = int(self.domain[0] / cell_size)
            n_y_cells = int(self.domain[1] / cell_size)
            grid = np.zeros((n_x_cells, n_y_cells), dtype=np.bool)

            # Setup to store results
            result = []

            # Assign all points to bins
            x_edges = np.linspace(0, self.domain[0], n_x_cells)
            y_edges = np.linspace(0, self.domain[1], n_y_cells)
            xbins = np.digitize(self.df['x'], bins=x_edges)
            ybins = np.digitize(self.df['y'], bins=y_edges)

            # Convert bins to cells
            xbins -= 1
            ybins -= 1
            pts = []
            for i, (idx, p) in enumerate(self.df.iterrows()):
                x_center = xbins[i]
                y_center = ybins[i]
                cell = grid[y_center, x_center]

                if cell == False:
                    result.append(idx)
                    pts.append((p[['x', 'y']]))
                    if len(result) > self.k + self.k * self.error_k:
                        # Too many points, break
                        min_idx = mid_idx
                        break

                    y_min = y_center - 5
                    if y_min < 0:
                        y_min = 0

                    x_min = x_center - 5
                    if x_min < 0:
                        x_min = 0

                    y_max = y_center + 5
                    if y_max > grid.shape[0]:
                        y_max = grid.shape[0]

                    x_max = x_center + 5
                    if x_max > grid.shape[1]:
                        x_max = grid.shape[1]

                    # Cover the necessary cells
                    grid[y_min: y_max,
                         x_min: x_max] = True

            #  Check break conditions
            if self.k - self.k * self.error_k <= len(result) <= self.k + self.k * self.error_k:
                break
            elif len(result) < self.k:
                # The radius is too large
                max_idx = mid_idx
            elif min_idx == mid_idx or mid_idx == max_idx:
                warnings.warn('Unable to optimally solve.  Returning with {} points'.format(len(result)))
                break

        self.mask = pd.Series(False, self.df.index)
        self.mask.loc[list(result)] = True
        state_package = {'mask': self.mask,
                         'k': self.k,
                         'error_k': self.error_k}

        self._action_stack.append(state_package)
        self._notify_subscribers(self)
        self._current_action_stack = len(self._action_stack) - 1  # 0 based vs. 1 based


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
    duplicate_mask = matches.duplicated(subset=['source_idx', 'destination_idx', 'distance'],
                                        keep='last')

    return duplicate_mask


def compute_fundamental_matrix(kp1, kp2, method='ransac', reproj_threshold=5.0, confidence=0.99):
    """
    Given two arrays of keypoints compute the fundamental matrix

    Parameters
    ----------
    kp1 : ndarray
          (n, 2) of coordinates from the source image

    kp2 : ndarray
          (n, 2) of coordinates from the destination image

    outlier_algorithm : {'ransac', 'lmeds', 'normal'}
                        The openCV algorithm to use for outlier detection

    reproj_threshold : float
                       The maximum distances in pixels a reprojected points
                       can be from the epipolar line to be considered an inlier

    confidence : float
                 [0, 1] that the estimated matrix is correct

    Returns
    -------
    transformation_matrix : ndarray
                            The 3x3 transformation matrix

    mask : ndarray
           Boolean array of the outliers

    Notes
    -----
    While the method is user definable, if the number of input points
    is < 7, normal outlier detection is automatically used, if 7 > n > 15,
    least medians is used, and if 7 > 15, ransac can be used.
    """

    if method == 'ransac':
        method_ = cv2.FM_RANSAC
    elif method == 'lmeds':
        method_ = cv2.FM_LMEDS
    elif method == 'normal':
        method_ = cv2.FM_7POINT
    else:
        raise ValueError("Unknown outlier detection method.  Choices are: 'ransac', 'lmeds', or 'normal'.")

    transformation_matrix, mask = cv2.findFundamentalMat(kp1,
                                                         kp2,
                                                         method_,
                                                         reproj_threshold,
                                                         confidence)
    try:
        mask = mask.astype(bool)
    except:
        pass  # pragma: no cover

    return transformation_matrix, mask


def compute_homography(kp1, kp2, method='ransac', **kwargs):
    """
    Given two arrays of keypoints compute the homography

    Parameters
    ----------
    kp1 : ndarray
          (n, 2) of coordinates from the source image

    kp2 : ndarray
          (n, 2) of coordinates from the destination image

    method : {'ransac', 'lmeds', 'normal'}
             The openCV algorithm to use for outlier detection

    ransacReprojThreshold : float
                            The maximum distances in pixels a reprojected points
                            can be from the epipolar line to be considered an inlier

    Returns
    -------
    transformation_matrix : ndarray
                            The 3x3 perspective transformation matrix

    mask : ndarray
           Boolean array of the outliers

    Notes
    -----
    While the method is user definable, if the number of input points
    is < 7, normal outlier detection is automatically used, if 7 > n > 15,
    least medians is used, and if 7 > 15, ransac can be used.
    """

    if method == 'ransac':
        method_ = cv2.RANSAC
    elif method == 'lmeds':
        method_ = cv2.LMEDS
    elif method == 'normal':
        method_ = 0  # Normal method
    else:
        raise ValueError("Unknown outlier detection method.  Choices are: 'ransac', 'lmeds', or 'normal'.")

    transformation_matrix, mask = cv2.findHomography(kp1,
                                                     kp2,
                                                     method_,
                                                     **kwargs)
    if mask is not None:
        mask = mask.astype(bool)
    return transformation_matrix, mask
