import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath('..'))

from .. import camera


class TestCamera(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.f = np.array([[3.27429392e-07, 8.31555374e-06, -1.37175583e-05],
                          [2.33349759e-07, 5.99315297e-07,  1.15196332e-01],
                          [-4.78065797e-03, -1.21448858e-01, 1.00000000e+00]])

    def test_compute_epipoles(self):
        e, e1 = camera.compute_epipoles(self.f)
        np.testing.assert_array_almost_equal(e, np.array([9.99999885e-01, -4.75272787e-04, 6.84672384e-05]))

        np.testing.assert_array_almost_equal(e1, np.array([[0.00000000e+00,  -6.84672384e-05,  -4.75272787e-04],
                                                    [6.84672384e-05,   0.00000000e+00,  -9.99999885e-01],
                                                    [4.75272787e-04,   9.99999885e-01,   0.00000000e+00]]))

    def test_idealized_camera(self):
        np.testing.assert_array_equal(np.eye(3,4), camera.idealized_camera())

    def test_estimated_camera_from_f(self):
        p1 = camera.estimated_camera_from_f(self.f)
        np.testing.assert_array_almost_equal(p1, np.array([[2.27210066e-06, 5.77212964e-05, -4.83159962e-04, 9.99999885e-01],
                                                    [4.78065745e-03,   1.21448845e-01,  -9.99999886e-01, -4.75272787e-04],
                                                    [2.33505351e-07,   6.03267385e-07,   1.15196312e-01, 6.84672384e-05]]))

    def test_triangulation_and_reprojection_error(self):
        p = np.eye(3,4)
        p1 = np.array([[2.27210066e-06, 5.77212964e-05, -4.83159962e-04, 9.99999885e-01],
                       [4.78065745e-03,   1.21448845e-01,  -9.99999886e-01, -4.75272787e-04],
                       [2.33505351e-07,   6.03267385e-07,   1.15196312e-01, 6.84672384e-05]])

        coords1 = np.array([[260.12573242, 6.37760448, 1.],
                            [539.05926514, 7.0553031 , 1.],
                            [465.07751465, 16.02966881, 1.],
                            [46.39139938, 16.96884346, 1.],
                            [456.28939819, 23.12134743, 1.]])

        coords2 = np.array([[707.23968506, 8.2479744, 1.],
                            [971.61566162, 18.7211895, 1.],
                            [905.67974854, 25.06698608, 1.],
                            [487.46420288, 11.28651524, 1.],
                            [897.73425293, 32.06435013, 1.]])

        truth = np.array([[3.03655751, 4.49076221, 4.17705934, 0.7984432 , 4.13673958],
                          [0.07447866, 0.05871216, 0.14393111, 0.29223435, 0.20962208],
                          [0.01167342, 0.00833074, 0.00898143, 0.01721084, 0.00906604],
                          [1., 1., 1., 1., 1. ]])

        c = camera.triangulate(coords1, coords2, p, p1)
        np.testing.assert_array_almost_equal(c, truth)

        truth = np.array([0.17603 ,  0.510191,  0.285109,  0.746513,  0.021731])
        residuals = camera.projection_error(p1, p, coords1.T, coords2.T)
        np.testing.assert_array_almost_equal(residuals, truth)
