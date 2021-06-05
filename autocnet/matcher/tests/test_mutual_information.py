import math
import os
import sys
import unittest
from unittest.mock import patch

import pytest
import numpy as np

from .. import mutual_information

def test_good_mi():
    test_image1 = np.array([[i for i in range(50)] for j in range(50)])
    # test_image2 = np.ones((50, 50))
    corrilation = mutual_information.mi(test_image1, test_image1)
    assert corrilation == 2.3025850929940455

def test_bad_mi():
    test_image1 = np.array([[i for i in range(50)] for j in range(50)])
    test_image2 = np.ones((50, 50))
    corrilation = mutual_information.mi(test_image1, test_image2)
    assert corrilation == pytest.approx(0)

def test_mutual_information():
    d_template = np.array([[i for i in range(50, 100)] for j in range(50)])
    s_image = np.ones((100, 100))

    s_image[25:75, 25:75] = d_template

    x_offset, y_offset, max_corr, corr_map = mutual_information.mutual_information(d_template, s_image, bins=20)
    assert x_offset == 0.9633527901853505
    assert y_offset == 0.5
    assert max_corr == 2.9755967600033015
    assert corr_map.shape == (51, 51)
    assert np.average(corr_map) == 1.3199548152066989
