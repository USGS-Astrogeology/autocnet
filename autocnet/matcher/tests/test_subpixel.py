import math
import os
import sys
import unittest
from unittest.mock import patch

import pytest

import numpy as np
from imageio import imread

from autocnet.examples import get_path
import autocnet.matcher.subpixel as sp


@pytest.fixture
def apollo_subsets():
    # These need to be geodata sets or just use mocks...
    arr1 = imread(get_path('AS15-M-0295_SML(1).png'))[100:201, 123:224]
    arr2 = imread(get_path('AS15-M-0295_SML(2).png'))[235:336, 95:196]
    return arr1, arr2

@pytest.mark.parametrize("nmatches, nstrengths", [(10,1), (10,2)])
def test_prep_subpixel(nmatches, nstrengths):
    arrs = sp._prep_subpixel(nmatches, nstrengths=nstrengths)
    assert len(arrs) == 5
    assert arrs[2].shape == (nmatches, nstrengths)
    assert arrs[0][0] == 0

@pytest.mark.parametrize("center_x, center_y, size, expected", [(4, 4, 9, 404),
                                                          (55.4, 63.1, 27, 6355)])
def test_clip_roi(center_x, center_y, size, expected):
    img = np.arange(10000).reshape(100, 100)

    clip, axr, ayr = sp.clip_roi(img, center_x, center_y, size)

    assert clip.mean() == expected

def test_subpixel_template(apollo_subsets):
    def clip_side_effect(*args, **kwargs):
        if np.array_equal(a, args[0]):
            return a, 0, 0
        else:
            center_y = b.shape[0] / 2
            center_x = b.shape[1] / 2
            bxr, bx = math.modf(center_x)
            byr, by = math.modf(center_y)
            bx = int(bx)
            by = int(by)
            return b[by-10:by+11, bx-10:bx+11], bxr, byr
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    with patch('autocnet.matcher.subpixel.clip_roi', side_effect=clip_side_effect):
        nx, ny, strength, _ = sp.subpixel_template(a.shape[1]/2, a.shape[0]/2,
                                                b.shape[1]/2, b.shape[0]/2,
                                                a, b, upsampling=16)

    assert strength >= 0.99
    assert nx == 50.5
    assert ny == 52.4375

@pytest.mark.parametrize("convergence_threshold, expected", [(2.0, (50.49, 52.08, (0.039507, -9.5e-20)))])
def test_iterative_phase(apollo_subsets, convergence_threshold, expected):
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    dx, dy, strength = sp.iterative_phase(a.shape[1]/2, a.shape[0]/2,
                                          b.shape[1]/2, b.shape[1]/2,
                                          a, b, 
                                          size=(51,51), 
                                          convergence_threshold=convergence_threshold,
                                          upsample_factor=100)
    assert dx == expected[0]
    assert dy == expected[1]
    if expected[2] is not None:
        for i in range(len(strength)):
            assert pytest.approx(strength[i],6) == expected[2][i]

@pytest.mark.parametrize("data, expected", [
    ((21,21), (10, 10)),
    ((20,20), (10,10))
])
def test_check_image_size(data, expected):
    assert sp.check_image_size(data) == expected

@pytest.mark.parametrize("x, y, x1, y1, image_size, template_size, expected",[
    (4, 3, 3, 2, (3,3), (3,3), (3,2)),
    (4, 3, 3, 2, (7,7), (3,3), (3,2)),  # Increase the search image size
    (4, 3, 3, 2, (7,7), (5,5), (3,2)), # Increase the template size
    (4, 3, 2, 2, (7,7), (3,3), (3,2)), # Move point in the x-axis
    (4, 3, 4, 3, (7,7), (3,3), (3,2)), # Move point in the other x-direction
    (4, 3, 3, 1, (7,7), (3,3), (3,2)), # Move point negative in the y-axis
    (4, 3, 3, 3, (7,7), (3,3), (3,2))  # Move point positive in the y-axis

])
def test_subpixel_template_cooked(x, y, x1, y1, image_size, template_size, expected):
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 1, 1, 1, 0, 1, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 1, 0, 1, 0, 0, 1, 0, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

    # Should yield (-3, 3) offset from image center
    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 1, 1, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

    dx, dy, corr, corrmap = sp.subpixel_template(x, y, x1, y1, 
                                                 test_image, t_shape,
                                                 image_size=image_size, 
                                                 template_size=template_size, 
                                                 upsampling=1)
    assert corr >= 1.0  # geq because sometime returning weird float > 1 from OpenCV
    assert dx == expected[0]
    assert dy == expected[1]

@pytest.mark.parametrize("x, y, x1, y1, image_size, expected",[
    (4, 3, 3, 2, (3,3), (3,2)),
    (4, 3, 3, 2, (5,5), (3,2)),  # Increase the search image size
    (4, 3, 3, 2, (5,5), (3,2)), # Increase the template size
    (4, 3, 2, 2, (5,5), (3,2)), # Move point in the x-axis
    (4, 3, 4, 3, (5,5), (3,2)), # Move point in the other x-direction
    (4, 3, 3, 1, (5,5), (3,2)), # Move point negative in the y-axis; also tests size reduction
    (4, 3, 3, 3, (5,5), (3,2))  # Move point positive in the y-axis

])
def test_subpixel_phase_cooked(x, y, x1, y1, image_size, expected):
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 1, 1, 1, 0, 1, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 1, 0, 1, 0, 0, 1, 0, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

    # Should yield (-3, 3) offset from image center
    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 1, 1, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

    dx, dy, metrics = sp.subpixel_phase(x, y, x1, y1, 
                                                 test_image, t_shape,
                                                 image_size=image_size)

    assert dx == expected[0]
    assert dy == expected[1]
