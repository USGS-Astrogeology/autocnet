import os
import numpy as np
import unittest
from autocnet.examples import get_path
import cv2

import sys

from .. import feature_extractor
from autocnet.fileio import io_gdal

sys.path.insert(0, os.path.abspath('..'))


class TestFeatureExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = io_gdal.GeoDataset(get_path('AS15-M-0295_SML.png'))
        cls.data_array = cls.dataset.read_array(dtype='uint8')
        cls.parameters = {"nfeatures": 10,
                          "nOctaveLayers": 3,
                          "contrastThreshold": 0.02,
                          "edgeThreshold": 10,
                          "sigma": 1.6}

    def test_extract_features(self):
        features = feature_extractor.extract_features(
            self.data_array, method='sift', extractor_parameters=self.parameters)
        self.assertEquals(len(features), 2)
        self.assertIn(len(features[0]), range(8, 12))
        self.assertIsInstance(features[0][0], type(cv2.KeyPoint()))
        self.assertIsInstance(features[1][0], np.ndarray)
