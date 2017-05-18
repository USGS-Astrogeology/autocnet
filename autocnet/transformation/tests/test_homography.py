import os
import sys
import unittest
import warnings
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

import autocnet.transformation.homography as hm

class TestHomography(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        # homography to transform fp
        static_H = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = np.empty(fph.shape)
        for i in range(nbr_inliers):
            proj = fph[i].dot(static_H.T)
            proj /= proj[-1]
            tp[i] = proj

        cls.H = static_H
        cls.fph = fph
        cls.tp = tp

    def test_Homography(self):

        H, mask = hm.compute_homography(self.fph, self.tp, method='lmeds')
        np.testing.assert_array_almost_equal(H, self.H)

        H, mask = hm.compute_homography(self.fph, self.tp, method='normal')
        np.testing.assert_array_almost_equal(H, self.H)

        H, mask = hm.compute_homography(self.fph, self.tp, method='ransac')
        np.testing.assert_array_almost_equal(H, self.H)

    def test_compute_error(self):
        error = hm.compute_error(self.H, self.fph, self.tp)
        np.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)

    def test_compute_error_nonhomogeneous(self):
        error = hm.compute_error(self.H, self.fph[:,:2], self.tp[:,:2])
        np.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)

    def test_compute_error_not_perfect(self):
        eps = np.random.normal(0,0.25, size=(3,3))
        print(eps)
        h = self.H + eps
        print(h)
        error = hm.compute_error(h, self.fph, self.tp)

        truth_means = np.array([ 0.5607765, 0.0027841, 1.64353546, 0.78280023])
        means = error.describe().loc['mean'].values
        np.testing.assert_array_almost_equal(truth_means, means)
