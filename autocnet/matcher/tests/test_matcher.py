import os
import sys
import unittest
import warnings

import cv2

from .. import feature
from autocnet.examples import get_path

sys.path.append(os.path.abspath('..'))


class TestMatcher(unittest.TestCase):

    def setUp(self):
        im1 = cv2.imread(get_path('AS15-M-0296_SML.png'))
        im2 = cv2.imread(get_path('AS15-M-0297_SML.png'))

        self.fd = {}

        sift = cv2.xfeatures2d.SIFT_create(10)

        self.fd['AS15-M-0296_SML.png'] = sift.detectAndCompute(im1, None)
        self.fd['AS15-M-0297_SML.png'] = sift.detectAndCompute(im2, None)

    def test_flann_match_k_eq_2(self):
        fmatcher = feature.FlannMatcher()
        source_image = self.fd['AS15-M-0296_SML.png']
        fmatcher.add(source_image[1], 0)

        self.assertTrue(len(fmatcher.nid_lookup), 1)

        fmatcher.train()

        with warnings.catch_warnings(record=True) as w:
            fmatcher.query(self.fd['AS15-M-0296_SML.png'][1], 0, k=2)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)

    def tearDown(self):
        pass
