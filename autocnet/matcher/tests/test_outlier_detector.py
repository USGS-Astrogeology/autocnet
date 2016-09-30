import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

from .. import outlier_detector
from autocnet.matcher.outlier_detector import SpatialSuppression

sys.path.append(os.path.abspath('..'))


class TestOutlierDetector(unittest.TestCase):

    def test_distance_ratio(self):
        df = pd.DataFrame(np.array([[0, 0, 1, 1, 2, 2, 2],
                                    [3, 4, 5, 6, 7, 8, 9],
                                    [1.25, 10.1, 2.3, 2.4, 1.2, 5.5, 5.7]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])

        d = outlier_detector.DistanceRatio(df)
        d.compute(single=True)
        self.assertEqual(d.nvalid, 2)

    def test_distance_ratio_unique(self):
        data = [['A', 0, 'B', 1, 10],
                ['A', 0, 'B', 8, 10]]
        df = pd.DataFrame(data, columns=['source_image', 'source_idx',
                                         'destination_image', 'destination_idx',
                                         'distance'])
        d = outlier_detector.DistanceRatio(df)
        d.compute(0.9)
        self.assertTrue(d.mask.all() == False)

    def test_mirroring_test(self):
        # returned mask should be same length as input df
        df = pd.DataFrame(np.array([[0, 0, 0, 1, 1, 1],
                                    [1, 2, 1, 1, 2, 3],
                                    [5, 2, 5, 5, 2, 3]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])
        mask = outlier_detector.mirroring_test(df)
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
        df = pd.DataFrame(data, columns=['x', 'y', 'strength'])
        self.suppression_obj = outlier_detector.SpatialSuppression(df, (100, 100), k=25)

    def test_properties(self):
        self.assertEqual(self.suppression_obj.k, 25)
        self.suppression_obj.k = 26
        self.assertTrue(self.suppression_obj.k, 26)

        self.assertEqual(self.suppression_obj.error_k, 0.1)
        self.suppression_obj.error_k = 0.05
        self.assertEqual(self.suppression_obj.error_k, 0.05)

        self.assertEqual(self.suppression_obj.nvalid, 0)
        self.assertIsInstance(self.suppression_obj.df, pd.DataFrame)

    def test_suppress_non_optimal(self):
        with warnings.catch_warnings(record=True) as w:
            self.suppression_obj.suppress()
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)

        self.assertEqual(self.suppression_obj.mask.sum(), 28)

    def test_suppress(self):
        self.suppression_obj.k = 30
        self.suppression_obj.suppress()
        self.assertIn(self.suppression_obj.mask.sum(), list(range(27, 35)))

        with warnings.catch_warnings(record=True) as w:
            self.suppression_obj.k = 101
            self.suppression_obj.suppress()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))

class testSuppressionRanges(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = np.random.RandomState(12345)

    def test_one_by_one(self):
        df = pd.DataFrame(self.r.uniform(0,1,(500, 3)), columns=['x', 'y', 'strength'])
        sup = SpatialSuppression(df, (1,1), k = 1)
        self.assertRaises(ValueError, sup.suppress())

    def test_min_max(self):
        df = pd.DataFrame(self.r.uniform(0,2,(500, 3)), columns=['x', 'y', 'strength'])
        sup = SpatialSuppression(df, (1.5,1.5), k = 1)
        sup.suppress()
        self.assertEqual(len(df[sup.mask]), 1)

    def test_point_overload(self):
        df = pd.DataFrame(self.r.uniform(0,15,(500, 3)), columns=['x', 'y', 'strength'])
        sup = SpatialSuppression(df, (15,15), k = 200)
        sup.suppress()
        self.assertEqual(len(df[sup.mask]), 70)

    def test_small_distribution(self):
        df = pd.DataFrame(self.r.uniform(0,25,(500, 3)), columns=['x', 'y', 'strength'])
        sup = SpatialSuppression(df, (25,25), k = 25)
        sup.suppress()
        self.assertEqual(len(df[sup.mask]), 28)

    def test_normal_distribution(self):
        df = pd.DataFrame(self.r.uniform(0,100,(500, 3)), columns=['x', 'y', 'strength'])
        sup = SpatialSuppression(df, (100,100), k = 15)
        sup.suppress()
        self.assertEqual(len(df[sup.mask]), 17)
