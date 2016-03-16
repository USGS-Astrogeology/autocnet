import os
import sys

import numpy as np
import unittest

from autocnet.examples import get_path
from autocnet.fileio.io_gdal import GeoDataset

from .. import node

sys.path.insert(0, os.path.abspath('..'))


class TestNode(unittest.TestCase):

    def setUp(self):
        img = get_path('AS15-M-0295_SML.png')
        self.node = node.Node(image_name='AS15-M-0295_SML',
                              image_path=img)

    def test_get_handle(self):
        self.assertIsInstance(self.node.handle, GeoDataset)

    def test_get_array(self):
        image = self.node.get_array()
        self.assertEqual((1012, 1012), image.shape)
        self.assertEqual(np.uint8, image.dtype)

    def test_extract_features(self):
        image = self.node.get_array()
        self.node.extract_features(
            image, extractor_parameters={
                'nfeatures': 10})
        self.assertEquals(len(self.node.keypoints), 10)
        self.assertEquals(len(self.node.descriptors), 10)
        self.assertIsInstance(self.node.descriptors[0], np.ndarray)
        self.assertEqual(10, self.node.nkeypoints)

    def test_convex_hull_ratio_fail(self):
        # Convex hull computation is checked lower in the hull computation
        self.assertRaises(AttributeError, self.node.coverage_ratio)

    def test_isis_serial(self):
        serial = self.node.isis_serial
        self.assertEqual(None, serial)
