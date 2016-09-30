import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import numpy.testing
import pandas as pd
from autocnet.transformation import transformations


class TestHomography(unittest.TestCase):

    def test_Homography(self):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        # homography to transform fp
        static_H = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = static_H.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1, :np.newaxis]
        H = transformations.Homography(static_H, index=np.arange(20))
        H.x1 = pd.DataFrame(fp, columns=['x', 'y'])
        H.x2 = pd.DataFrame(tp.T[:, :2], columns=['x', 'y'])
        self.assertAlmostEqual(H.determinant, 0.6249999, 5)
        self.assertAlmostEqual(H.condition, 7.19064438, 5)

        error = H.error
        numpy.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)

        description = H.describe_error
        self.assertIsInstance(description, pd.DataFrame)

    def test_Homography_fail(self):
        with self.assertRaises(TypeError):
            transformations.Homography([1,2,3], np.arange(3), np.arange(3), None)


class TestFundamentalMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        static_F = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = static_F.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1, :np.newaxis]

        cls.F = transformations.FundamentalMatrix(static_F, index=np.arange(20))
        cls.F.x1 = pd.DataFrame(fph, columns=['x', 'y', 'h'])
        cls.F.x2 = pd.DataFrame(tp.T, columns=['x', 'y', 'h'])

    def test_compute_f(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))
        tp = np.array(np.random.standard_normal((nbr_inliers, 2)))

        F = transformations.FundamentalMatrix(np.zeros((3,3)), index=np.arange(20))
        F.compute(fp, tp)

        np.testing.assert_array_almost_equal(F, np.array([[-0.685892, -5.870193, 2.268333],
                                                          [-0.704199, 12.88776,  -3.040341],
                                                          [-0.231815, -2.806056, 1.]]))

    def test_f_error(self):
        self.assertIsInstance(self.F.error, pd.Series)

    def test_f_determinant(self):
        self.assertAlmostEqual(self.F.determinant, 0.624999, 5)

    def test_f_rank(self):
        # Degenerate Case
        self.assertEqual(self.F.rank, 3)

    def test_f_refine(self):
        # This should raise an error.
        self.F.refine()
        self.assertEqual(len(self.F._action_stack), 2)

        # Previous error should preclude do/undo
        self.F.rollback()
        self.assertEqual(self.F._current_action_stack, 0)
        self.F.rollforward()
        self.assertEqual(self.F._current_action_stack, 1)

        self.F._clean_attrs()
        self.assertNotIn('_error', self.F.__dict__)