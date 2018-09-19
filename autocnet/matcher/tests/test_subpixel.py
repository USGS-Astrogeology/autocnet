import os
import sys
import unittest

import pytest

import numpy as np
from imageio import imread

from autocnet.examples import get_path
import autocnet.matcher.subpixel as sp


@pytest.fixture
def apollo_subsets():
    arr1 = imread(get_path('AS15-M-0295_SML(1).png'))[100:200, 123:223]
    arr2 = imread(get_path('AS15-M-0295_SML(2).png'))[235:335, 95:195]
    return arr1, arr2

@pytest.mark.parametrize("nmatches, nstrengths", [(10,1), (10,2)])
def test_prep_subpixel(nmatches, nstrengths):
    arrs = sp._prep_subpixel(nmatches, nstrengths=nstrengths)
    assert len(arrs) == 5
    assert arrs[2].shape == (nmatches, nstrengths)
    assert np.isnan(arrs[0][0])

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
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    midy = int(b.shape[0] / 2)
    midx = int(b.shape[1] / 2)
    subb = b[midy-10:midy+10, midx-10:midx+10]
    xoff, yoff, err = sp.subpixel_template(subb, a)
    assert xoff == 0.0625 
    assert yoff == 2.125 
    assert err == 0.9905822277069092
