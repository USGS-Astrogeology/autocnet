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


def test_subpixel_phase(apollo_subsets):
    a = apollo_subsets[0]
    b = apollo_subsets[1]

    xoff, yoff, err = sp.subpixel_phase(a, b)
    assert xoff == 0
    assert yoff == 2
    assert len(err) == 2

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
        nx, ny, strength = sp.subpixel_template(a.shape[1]/2, a.shape[0]/2,
                                                b.shape[1]/2, b.shape[0]/2,
                                                a, b, upsampling=16)
    
    assert strength >= 0.99
    assert nx == 50.9375
    assert ny == 48.9375

@pytest.mark.parametrize("convergence_threshold, expected", [(1.0, (None, None, None)),
                                                             (2.0, (50.49, 52.44, (0.039507, -9.5e-20)))])
def test_iterative_phase(apollo_subsets, convergence_threshold, expected):
    def clip_side_effect(*args, **kwargs):
        if np.array_equal(a, args[0]):
            return a, 0, 0
        else:
            return b, 0, 0
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    with patch('autocnet.matcher.subpixel.clip_roi', side_effect=clip_side_effect):
        nx, ny, strength = sp.iterative_phase(a.shape[1]/2, a.shape[0]/2,
                                              b.shape[1]/2, b.shape[1]/2,
                                              a, b, convergence_threshold=convergence_threshold,
                                              upsample_factor=100)
        assert nx == expected[0]
        assert ny == expected[1]
        if expected[2] is not None:
            for i in range(len(strength)):
                assert pytest.approx(strength[i],6) == expected[2][i]

@pytest.mark.parametrize("data, expected", [
    ((21,21), (10.5, 10.5)),
    ((20,20), (11,11)),
    ((0,0), (1,1))
])
def test_check_image_size(data, expected):
    assert sp.check_image_size(data) == expected