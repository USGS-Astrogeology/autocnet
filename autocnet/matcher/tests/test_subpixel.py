import os
import sys
import unittest

import numpy as np

from .. import subpixel as sp

sys.path.append(os.path.abspath('..'))


class TestSubPixel(unittest.TestCase):

    def setup(self):
        pass

    def test_clip_roi(self):
        img = np.arange(10000).reshape(100, 100)
        center = (4, 4)

        clip = sp.clip_roi(img, center, 9)
        self.assertEqual(clip.mean(), 404)

        center = (55.4, 63.1)
        clip = sp.clip_roi(img, center, 27)
        self.assertEqual(clip.mean(), 6355.0)

        self.assertRaises(ValueError, sp.clip_roi, img, center, 10)
