import os
import sys
import unittest
import warnings
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

import autocnet.transformation.fundamental_matrix as fm

class TestFundamentalMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nbr_inliers = 20
        np.random.seed(12345)
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))  # inliers

        static_F = np.array([[4, 0.5, 10], [0.25, 1, 5], [0.2, 0.1, 1]])

        # Make homogeneous
        fph = np.hstack((fp, np.ones((nbr_inliers, 1))))
        tp = np.empty((nbr_inliers, 3))
        for i, j in enumerate(fph):
            proj = j.dot(static_F)
            proj /= proj[2]
            tp[i] = proj

        cls.static_F = static_F
        cls.F = np.array([[-0.685892, -5.870193, 2.268333],
                          [-0.704199, 12.88776,  -3.040341],
                          [-0.231815, -2.806056, 1.]])

        cls.x1 = pd.DataFrame(fph, columns=['x', 'y', 'h'])
        cls.x2 = pd.DataFrame(tp, columns=['x', 'y', 'h'])

        cls.fixed_x1 = np.array([[ 438.394104  ,  846.43518066,    1.        ],
       [ 767.89105225,  380.79367065,    1.],
       [  63.80842972,  815.14257812,    1.        ],
       [ 283.96408081,  901.07287598,    1.        ],
       [ 421.63833618,  841.66619873,    1.        ],
       [ 181.8278656 ,  706.01611328,    1.        ],
       [ 650.27160645,  416.72653198,    1.        ],
       [ 650.27160645,  416.72653198,    1.        ],
       [ 721.18585205,  368.2802124 ,    1.        ],
       [  88.97966003,  962.11322021,    1.        ]])
        cls.fixed_x2 = np.array([[ 652.32714844,  847.51605225,    1.        ],
       [ 985.95928955,  384.58950806,    1.        ],
       [ 281.5947876 ,  819.9956665 ,    1.        ],
       [ 501.13912964,  904.06054688,    1.        ],
       [ 637.31488037,  842.93652344,    1.        ],
       [ 398.35501099,  708.86029053,    1.        ],
       [ 875.11975098,  419.54541016,    1.        ],
       [ 875.11975098,  419.54541016,    1.        ],
       [ 943.69946289,  372.12527466,    1.        ],
       [ 299.27636719,  968.44104004,    1.        ]])
        cls.fixed_f = np.array([[ -2.85373973e-08,   3.02728824e-06,  -2.41915056e-03],
       [ -4.53237187e-06,   1.38905788e-07,  -4.14644099e-02],
       [  3.25687216e-03,   4.11777575e-02,   3.61272746e-01]])

    def test_compute_f(self):
        # The F matrix is good if the sum of the error is within some threshold.
        F, mask = fm.compute_fundamental_matrix(self.x1, self.x2, method='ransac')
        self.assertTrue(abs(sum(fm.compute_fundamental_error(F, self.x1, self.x2))) < 0.01)

        F, mask = fm.compute_fundamental_matrix(self.x1, self.x2, method='lmeds')
        self.assertTrue(abs(sum(fm.compute_fundamental_error(F, self.x1, self.x2))) < 0.01)

        F, mask = fm.compute_fundamental_matrix(self.x1, self.x2, method='normal')
        self.assertTrue(abs(sum(fm.compute_fundamental_error(F, self.x1, self.x2))) < 0.01)

        F, mask = fm.compute_fundamental_matrix(self.x1, self.x2, method='8point')
        self.assertTrue(abs(sum(fm.compute_fundamental_error(F, self.x1, self.x2))) < 0.01)

        F, mask = fm.compute_fundamental_matrix(self.x1, self.x2, method='mle')
        self.assertTrue(abs(sum(fm.compute_fundamental_error(F, self.x1, self.x2))) < 0.01)

    def test_compute_mle_f(self):
        #TODO: Write a better test for MLE the data here is too clean.
        pass

    def test_f_reprojection_error(self):
        err = fm.compute_reprojection_error(self.fixed_f,
                                            self.fixed_x1,
                                            self.fixed_x2)
        self.assertTrue(err.mean() < 0.5)

    def test_f_fundamental_error(self):
        err = fm.compute_fundamental_error(self.fixed_f,
                                           self.fixed_x1,
                                           self.fixed_x2)
        self.assertTrue(abs(sum(err)) < 0.03)

    def test_update_fundamental_mask(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))
        tp = np.array(np.random.standard_normal((nbr_inliers, 2)))

        F, mask = fm.compute_fundamental_matrix(fp, tp, method='ransac')

        new_mask = fm.update_fundamental_mask(F, fp, tp, threshold=0.5, method='reprojection')
        self.assertEqual(10, new_mask['fundamental'].sum())

    def test_update_fundamental_mask_with_index(self):
        np.random.seed(12345)
        nbr_inliers = 20
        fp = np.array(np.random.standard_normal((nbr_inliers, 2)))
        tp = np.array(np.random.standard_normal((nbr_inliers, 2)))

        F, mask = fm.compute_fundamental_matrix(fp, tp, method='ransac')
        new_index = np.arange(20)[::-1]  #Just reverse the index
        new_mask = fm.update_fundamental_mask(F, fp, tp, threshold=0.5, index=new_index)
        np.testing.assert_array_equal(new_index, new_mask.index.values)

    def test_update_fundamental_mask_with_fundamental(self):
        new_mask = fm.update_fundamental_mask(self.fixed_f,
                                              self.fixed_x1,
                                              self.fixed_x2,
                                              threshold=0.05,
                                              method='fundamental')

        self.assertTrue(new_mask['fundamental'].sum() == 10)
        new_mask = fm.update_fundamental_mask(self.fixed_f,
                                              self.fixed_x1,
                                              self.fixed_x2,
                                              threshold=0.005,
                                              method='fundamental')
        print(new_mask['fundamental'].sum())

        self.assertTrue(new_mask['fundamental'].sum() == 9)

    def test_enforce_singularity_constraint(self):
        r3 = np.array([[1, 0, 1],[-2, -3, 1],[2, -3, 1]])
        F = fm.enforce_singularity_constraint(r3)
        self.assertEqual(2, np.linalg.matrix_rank(F))
