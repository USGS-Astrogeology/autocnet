import pytest
import numpy as np
import autocnet.camera.camera as cam

@pytest.fixture
def F():
    return np.array([[-2.62542734e-07, 3.24680544e-08, 1.06264873e-06],
                     [-2.58739695e-07, 3.30340456e-09, 0.0015042137],
                     [0.0134032928, -0.00167767917, 1.00000026]])

def test_compute_epipoles(F):
        e, e1 = cam.compute_epipoles(F)
        np.testing.assert_array_almost_equal(e, np.array([ 9.99908150e-01, -1.35532934e-02, 1.93244935e-05]))

        np.testing.assert_array_almost_equal(e1, 
                                             np.array([[  0.00000000e+00, 1.93244935e-05, 1.35532934e-02],
                                                       [ -1.93244935e-05, 0.00000000e+00, 9.99908150e-01],
                                                       [ -1.35532934e-02, -9.99908150e-01, 0.00000000e+00]]))

def test_camera_from_f(F):
    truth = np.array([[-1.81658755e-04,  2.27380780e-05, -1.35533260e-02, 9.99908150e-01],
                      [-1.34020617e-02,  1.67752508e-03, -9.99908410e-01, -1.35532934e-02],
                      [-2.62274248e-07,  3.74315021e-09,  1.50408994e-03, 1.93244935e-05]])
    p = cam.camera_from_f(F)
    np.testing.assert_almost_equal(p, truth)

def test_idealized_camera():
    np.testing.assert_array_equal(np.eye(3,4), cam.idealized_camera())

def test_triangulation_and_reprojection_error():
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

    c = cam.triangulate(coords1, coords2, p, p1)
    np.testing.assert_array_almost_equal(c, truth)

    truth = np.array([0.17603 ,  0.510191,  0.285109,  0.746513,  0.021731])
    residuals = cam.projection_error(p1, p, coords1.T, coords2.T)
    np.testing.assert_array_almost_equal(residuals, truth)