import os
import sys
import unittest

import cv2

sys.path.append(os.path.abspath('..'))

from .. import matcher
from autocnet.examples import get_path


class TestMatcher(unittest.TestCase):

    def setUp(self):
        im1 = cv2.imread(get_path('AS15-M-0296_SML.png'))
        im2 = cv2.imread(get_path('AS15-M-0297_SML.png'))

        self.fd = {}

        sift = cv2.xfeatures2d.SIFT_create(10)

        self.fd['AS15-M-0296_SML.png'] = sift.detectAndCompute(im1, None)
        self.fd['AS15-M-0297_SML.png'] = sift.detectAndCompute(im2, None)

    def test_flann_match_k_eq_2(self):
        fmatcher = matcher.FlannMatcher()
        truth_image_indices = {}
        counter = 0
        for imageid, (keypoint, descriptor) in self.fd.items():
            truth_image_indices[counter] = imageid
            fmatcher.add(descriptor, imageid)
            counter += 1

        self.assertEqual(truth_image_indices, fmatcher.image_indices)

        fmatcher.train()

        matched = fmatcher.query(self.fd['AS15-M-0296_SML.png'][1], k=2)
        self.assertEqual(10, len(matched))

        # Check that self neighbors are being omitted
        distance = matched['distance']
        self.assertFalse(distance[distance == 0].any())

    def test_flann_match_k_eq_3(self):
        fmatcher = matcher.FlannMatcher()
        truth_image_indices = {}
        counter = 0
        for imageid, (keypoint, descriptor) in self.fd.items():
            truth_image_indices[counter] = imageid
            fmatcher.add(descriptor, imageid)
            counter += 1

        self.assertEqual(truth_image_indices, fmatcher.image_indices)

        fmatcher.train()

        matched = fmatcher.query(self.fd['AS15-M-0296_SML.png'][1], k=3)
        self.assertEqual(20, len(matched))

        # Check that self neighbors are being omitted
        distance = matched['distance']
        self.assertFalse(distance[distance == 0].any())

    def tearDown(self):
        pass