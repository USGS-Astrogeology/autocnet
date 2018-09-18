import os
import numpy as np
import pandas as pd
import unittest
from autocnet.examples import get_path
import cv2

import sys

from .. import cpu_extractor
from plio.io import io_gdal


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

    def test_extract_vlfeat(self):
        kps, descriptors = cpu_extractor.extract_features(self.data_array,
                                                              extractor_method='vlfeat',
                                                              extractor_parameters={})
        self.assertIsInstance(kps, pd.DataFrame)
        self.assertEqual(descriptors.dtype, np.float32)

    def test_extract_vlfeat_without(self):
        cpu_extractor.vlfeat = False
        with self.assertRaises(ImportError):
            kps, descriptors = cpu_extractor.extract_features(self.data_array,
                                                                  extractor_method='vlfeat',
                                                                  extractor_parameters={})
        cpu_extractor.vlfeat = True
