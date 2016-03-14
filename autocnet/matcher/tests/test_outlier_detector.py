import os
import sys
import unittest
import warnings

# import cv2
import numpy as np
import pandas as pd

from .. import outlier_detector
sys.path.append(os.path.abspath('..'))


class TestOutlierDetector(unittest.TestCase):

    def test_distance_ratio(self):
        df = pd.DataFrame(np.array([[0, 0, 1, 1, 2, 2, 2],
                                    [3, 4, 5, 6, 7, 8, 9],
                                    [1.25, 10.1, 2.3, 2.4, 1.2, 5.5, 5.7]]).T,
                          columns=['source_idx', 'destination_idx', 'distance'])
        d = outlier_detector.DistanceRatio(df)
        d.compute()
        self.assertEqual(d.nvalid, 2)

    def test_distance_ratio_unique(self):
        data = [['A', 0, 'B', 1, 10],
                ['A', 0, 'B', 8, 10]]
        df = pd.DataFrame(
            data,
            columns=[
                'source_image',
                'source_idx',
                'destination_image',
                'destination_idx',
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

    def test_compute_fundamental_matrix(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))
        tp = np.array(np.random.standard_normal((nbr_inliers, 2)))

        F, mask = outlier_detector.compute_fundamental_matrix(
            fp, tp, confidence=0.5)

        np.testing.assert_array_almost_equal(F, np.array([[-0.53516611, 2.34420116, -0.60565672],
                                                          [-0.08070418, -2.77970059, 1.99678886],
                                                          [-0.89519184, 0.90058511, 1.]]))

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
        self.suppression_obj = outlier_detector.SpatialSuppression(
            df, (100, 100), k=25)

    def test_properties(self):
        self.assertEqual(self.suppression_obj.k, 25)
        self.suppression_obj.k = 26
        self.assertTrue(self.suppression_obj.k, 26)

        self.assertEqual(self.suppression_obj.error_k, 0.1)
        self.suppression_obj.error_k = 0.05
        self.assertEqual(self.suppression_obj.error_k, 0.05)

        self.assertEqual(self.suppression_obj.nvalid, None)
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
        self.assertIn(self.suppression_obj.mask.sum(), list(range(27, 34)))
