import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

from .. import cpu_outlier_detector

sys.path.append(os.path.abspath('..'))


class TestOutlierDetector(unittest.TestCase):

    def test_distance_ratio(self):
        df = pd.DataFrame(np.array([[0, 0, 1, 1, 2, 2, 2],
                                    [3, 4, 5, 6, 7, 8, 9],
                                    [1.25, 10.1, 2.3, 2.4, 1.2, 5.5, 5.7]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])

        mask = cpu_outlier_detector.distance_ratio(None, df, ratio=0.8)
        self.assertEqual(2, sum(mask))

    def test_distance_ratio_unique(self):
        data = [['A', 0, 'B', 1, 10],
                ['A', 0, 'B', 8, 10]]
        df = pd.DataFrame(data, columns=['source_image', 'source_idx',
                                         'destination_image', 'destination_idx',
                                         'distance'])
        mask = cpu_outlier_detector.distance_ratio(None, df, ratio=0.9)
        self.assertTrue(mask.all() == False)

    def test_mirroring_test(self):
        # returned mask should be same length as input df
        df = pd.DataFrame(np.array([[0, 0, 0, 1, 1, 1],
                                    [1, 2, 1, 1, 2, 3],
                                    [5, 2, 5, 5, 2, 3]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])
        mask = cpu_outlier_detector.mirroring_test(df)
        self.assertEqual(mask.sum(), 1)

    def tearDown(self):
        pass


class TestSpatialSuppression(unittest.TestCase):

    def setUp(self):
        seed = np.random.RandomState(12345)
        x = seed.randint(0, 100, 100).astype(np.float32)
        y = seed.randint(0, 100, 100).astype(np.float32)
        strength = seed.rand(100)
        data = np.vstack((x, y, strength)).T
        self.df = pd.DataFrame(data, columns=['lon', 'lat', 'strength'])
        self.domain = (0,0, 100, 100)

    def test_suppress(self):
        mask, k = cpu_outlier_detector.spatial_suppression(self.df, self.domain, k=25)
        self.assertEqual(mask.sum(), 25)

        mask, k = cpu_outlier_detector.spatial_suppression(self.df, self.domain, k=30, error_k=.15)
        self.assertEqual(mask.sum(), 34)

    def test_suppress_non_optimal(self):
        with warnings.catch_warnings(record=True) as w:
            mask, k = cpu_outlier_detector.spatial_suppression(self.df, self.domain, k=30)
            self.assertEqual(len(w), 1)
            self.assertEqual(k, 59)
            self.assertTrue(issubclass(w[0].category, UserWarning))

class testSuppressionRanges(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = np.random.RandomState(12345)

    def test_min_max(self):
        df = pd.DataFrame(self.r.uniform(0,2,(500, 3)), columns=['lon', 'lat', 'strength'])
        mask, k = cpu_outlier_detector.spatial_suppression(df, (0, 0, 1.5, 1.5), k = 1)
        self.assertEqual(len(df[mask]), 1)

    def test_point_overload(self):
        df = pd.DataFrame(self.r.uniform(0,15,(500, 3)), columns=['lon', 'lat', 'strength'])
        mask, k = cpu_outlier_detector.spatial_suppression(df, (0, 0, 15, 15), k = 200)
        self.assertEqual(len(df[mask]), 195)

    def test_small_distribution(self):
        df = pd.DataFrame(self.r.uniform(0,25,(500, 3)), columns=['x', 'y', 'strength'])
        mask, k = cpu_outlier_detector.spatial_suppression(df, (0, 0, 25, 25), k = 25, xkey='x', ykey='y')
        self.assertEqual(len(df[mask]), 25)

    def test_normal_distribution(self):
        df = pd.DataFrame(self.r.uniform(0,100,(500, 3)), columns=['x', 'y', 'strength'])
        mask, k = cpu_outlier_detector.spatial_suppression(df, (0, 0, 100, 100), k = 15, xkey='x', ykey='y')
        self.assertEqual(len(df[mask]), 17)
