import os
import sys

import unittest
import warnings

import numpy as np
import pandas as pd

from autocnet.examples import get_path
from plio.io.io_gdal import GeoDataset

from .. import node

sys.path.insert(0, os.path.abspath('..'))


class TestNode(unittest.TestCase):

    def setUp(self):
        img = get_path('AS15-M-0295_SML.png')
        self.node = node.Node(image_name='AS15-M-0295_SML',
                              image_path=img)

    def test_get_handle(self):
        self.assertIsInstance(self.node.geodata, GeoDataset)

    def test_get_array(self):
        image = self.node.get_array()
        self.assertEqual((1012, 1012), image.shape)
        self.assertEqual(np.uint8, image.dtype)

    def test_extract_features(self):
        image = self.node.get_array()
        self.node.extract_features(image, extractor_parameters={'nfeatures': 10})
        self.assertEquals(len(self.node.get_keypoints()), 10)
        self.assertEquals(len(self.node.descriptors), 10)
        self.assertIsInstance(self.node.descriptors[0], np.ndarray)
        self.assertEqual(10, self.node.nkeypoints)

    def test_masks(self):
        # Assert a warning raise here
        with warnings.catch_warnings(record=True) as w:
            masks = self.node.masks
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)

        image = self.node.get_array()
        self.node.extract_features(image, extractor_parameters={'nfeatures': 5})
        self.assertIsInstance(self.node.masks, pd.DataFrame)
        # Create an artificial mask
        self.node.masks = ('foo', np.array([0, 0, 1, 1, 1], dtype=np.bool))
        self.assertEqual(self.node.masks['foo'].sum(), 3)

    def test_convex_hull_ratio_fail(self):
        # Convex hull computation is checked lower in the hull computation
        self.assertRaises(AttributeError, self.node.coverage_ratio)

    def test_isis_serial(self):
        serial = self.node.isis_serial
        self.assertEqual(None, serial)

    def test_save_load(self):
        image = self.node.get_array()
        self.node.extract_features(image, method='sift', extractor_parameters={'nfeatures': 10})
        self.node.save_features('node_test.hdf')
        kps = self.node.get_keypoints().copy()
        descriptors = self.node.descriptors.copy()
        self.node.load_features('node_test.hdf')

        self.assertTrue((kps.sort(axis=0) == self.node.get_keypoints().sort(axis=0)).all().all())

        os.remove('node_test.hdf')

    def test_coverage(self):
        image = self.node.get_array()
        self.node.extract_features(image, method='sift', extractor_parameters={'nfeatures': 10})

        coverage_percn = self.node.coverage()

        self.assertAlmostEqual(coverage_percn, 38.06139557)
