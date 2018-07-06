import os
import sys
import unittest

import pytest

import numpy as np
from scipy.ndimage import imread

from autocnet.examples import get_path
import autocnet.matcher.subpixel as sp


@pytest.fixture
def apollo_subsets():
    arr1 = imread(get_path('AS15-M-0295_SML(1).png'))[100:200, 123:223]
    arr2 = imread(get_path('AS15-M-0295_SML(2).png'))[235:335, 95:195]
    print(arr1.shape, arr2.shape)
    return arr1, arr2

def test_clip_roi():
    img = np.arange(10000).reshape(100, 100)
    center = (4, 4)

    clip = sp.clip_roi(img, center, 9)
    assert clip.mean() == 404

    center = (55.4, 63.1)
    clip = sp.clip_roi(img, center, 27)
    assert clip.mean() == 6355.0

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
    xoff, yoff, err = sp.subpixel_offset(subb, a)
    assert xoff == 0.0625 
    assert yoff == 2.125 
    assert err == 0.9905822277069092
