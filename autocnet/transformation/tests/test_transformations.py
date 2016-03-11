import os
import sys
import unittest
# import warnings

import numpy as np
import numpy.testing
import pandas as pd
from autocnet.transformation import transformations

sys.path.insert(0, os.path.abspath('..'))


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
        H = transformations.Homography(
            static_H, pd.DataFrame(
                fp, columns=[
                    'x', 'y']), pd.DataFrame(
                tp.T[
                    :, :2], columns=[
                        'x', 'y']), mask=pd.Series(
                            True, index=np.arange(
                                fp.shape[0])))
        self.assertAlmostEqual(H.determinant, 0.6249999, 5)
        self.assertAlmostEqual(H.condition, 7.19064438, 5)

        error = H.error
        numpy.testing.assert_array_almost_equal(error['rmse'], np.zeros(20))
        self.assertAlmostEqual(error.total_rms, 0.0, 1)
        self.assertAlmostEqual(error.x_rms, 0.0, 1)
        self.assertAlmostEqual(error.y_rms, 0.0, 1)

    def test_Homography_fail(self):
        self.assertRaises(
            TypeError, transformations.Homography, [
                1, 2, 3], 'a', 'b')


class TestFundamentalMatrix(unittest.TestCase):

    def test_FundamentalMatrix(self):
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        static_F = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = static_F.dot(fph.T)
        # normalize hom. coordinates
        tp /= tp[-1, :np.newaxis]

        F = transformations.FundamentalMatrix(
            static_F, pd.DataFrame(
                fp, columns=[
                    'x', 'y']), pd.DataFrame(
                tp.T[
                    :, :2], columns=[
                        'x', 'y']), mask=pd.Series(
                            True, index=np.arange(
                                fp.shape[0])))

        self.assertAlmostEqual(F.determinant, 0.624999, 5)

        self.assertIsInstance(F.error, pd.DataFrame)

        # This should raise an error.
        F.refine()
        self.assertIsInstance(F.error, pd.DataFrame)
        self.assertEqual(len(F._action_stack), 1)

        # Previous error should preclude do/undo
        F.rollback()
        self.assertEqual(F._current_action_stack, 0)
        F.rollforward()
        self.assertEqual(F._current_action_stack, 0)

        F._clean_attrs()
        self.assertNotIn('_error', F.__dict__)
