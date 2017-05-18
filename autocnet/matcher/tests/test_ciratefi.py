import math
import unittest
import warnings

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate

from autocnet.examples import get_path
from autocnet.matcher import subpixel as sp
from .. import ciratefi


class TestCiratefi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        img = imread(get_path('AS15-M-0298_SML.png'), flatten=True)
        img_coord = (482.09783936, 652.40679932)

        cls.template = sp.clip_roi(img, img_coord, 5)
        cls.template = rotate(cls.template, 90)
        cls.template = imresize(cls.template, 1.)

        cls.search = sp.clip_roi(img, img_coord, 21)
        cls.search = rotate(cls.search, 0)
        cls.search = imresize(cls.search, 1.)

        cls.offset = (1, 1)

        cls.offset_template = sp.clip_roi(img, np.add(img_coord, cls.offset), 5)
        cls.offset_template = rotate(cls.offset_template, 0)
        cls.offset_template = imresize(cls.offset_template, 1.)

        cls.search_center = [math.floor(cls.search.shape[0]/2),
                             math.floor(cls.search.shape[1]/2)]

        cls.upsampling = 10
        cls.alpha = math.pi/2
        cls.cifi_thresh = 90
        cls.rafi_thresh = 90
        cls.tefi_thresh = 100
        cls.use_percentile = True
        cls.radii = list(range(1, 3))

        cls.cifi_number_of_warnings = 2
        cls.rafi_number_of_warnings = 2

    def test_cifi(self):
            # check all warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ciratefi.cifi(self.template, self.search, 1.0, radii=[100], use_percentile=False)
                self.assertEqual(len(w), self.cifi_number_of_warnings)

            # Threshold out of bounds error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, -1.1, use_percentile=False)
            # radii list empty/none error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, 90, radii=None)
            # scales list empty/none error
            self.assertRaises(ValueError, ciratefi.cifi, self.template, self.search, 90, scales=None)
            # template is bigger than search error
            self.assertRaises(ValueError, ciratefi.cifi, self.search, self.template, -1.1, use_percentile=False)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pixels, scales = ciratefi.cifi(self.template, self.search, thresh=self.cifi_thresh,
                                               radii=self.radii, use_percentile=True)

                self.assertEqual(len(w), 0)

                self.assertEqual(self.search.shape, scales.shape)
                self.assertIn((np.floor(self.search.shape[0]/2), np.floor(self.search.shape[1]/2)), pixels)
                self.assertTrue(pixels.size in range(0, self.search.size))

    def test_rafi(self):
        rafi_pixels = [(10, 10)]

        rafi_scales = np.ones(self.search.shape, dtype=float)

        # check all warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ciratefi.rafi(self.template, self.search, rafi_pixels,
                          rafi_scales, thresh=1, radii=[100],
                          use_percentile=False)

            self.assertEqual(len(w), self.rafi_number_of_warnings)

        # Threshold out of bounds error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, rafi_scales,
                          -1.1, use_percentile=False)

        # Radii list is empty.None error
        self.assertRaises(ValueError, ciratefi.rafi, self.search, self.template, rafi_pixels, -1.1, radii=None)
        # candidate pixel list empty/none error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, [], rafi_scales)
        # scales list empty/none error
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, None)
        # template is bigger than search error
        self.assertRaises(ValueError, ciratefi.rafi, self.search, self.template, rafi_pixels, rafi_scales)
        # best scale nd array is not equal image shape
        self.assertRaises(ValueError, ciratefi.rafi, self.template, self.search, rafi_pixels, rafi_scales[:10])

        with warnings.catch_warnings(record=True) as w:
            pixels, scales = ciratefi.rafi(self.template, self.search, rafi_pixels, rafi_scales,
                                           thresh=self.rafi_thresh, radii=self.radii, use_percentile=True,
                                           alpha=self.alpha)
            self.assertEqual(len(w), 0)

            self.assertIn((np.floor(self.search.shape[0]/2), np.floor(self.search.shape[1]/2)), pixels)
            self.assertTrue(pixels.size in range(0, self.search.size))

    def test_tefi(self):
        tefi_pixels = [(10, 10)]
        tefi_scales = np.ones(self.search.shape, dtype=float)
        tefi_angles = [3.14159265]

        # Threshold out of bounds error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, tefi_scales,
                          tefi_angles, thresh=-1.1, use_percentile=False, alpha=self.alpha)

        # angle list is empty/None error
        self.assertRaises(ValueError, ciratefi.tefi, self.search, self.template, tefi_pixels, tefi_scales, None)
        # candidate pixel list empty/none error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, None, tefi_scales, tefi_angles)
        # scales list empty/none error
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, None, tefi_angles)
        # template is bigger than search error
        self.assertRaises(ValueError, ciratefi.tefi, self.search, self.template, tefi_pixels, tefi_scales, -1.1)
        # best scale nd array is smaller than number of candidate pixels
        self.assertRaises(ValueError, ciratefi.tefi, self.template, self.search, tefi_pixels, tefi_scales[:10], -1.1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pixel = ciratefi.tefi(self.template, self.search, tefi_pixels, tefi_scales, tefi_angles,
                                           thresh=self.tefi_thresh, use_percentile=True, alpha=self.alpha,
                                           upsampling=self.upsampling)

            for warn in w:
                print(warn)

            self.assertEqual(len(w), 0)
            print(pixel)
            self.assertTrue(np.equal((.5, .5), (pixel[1], pixel[0])).all())

    def test_ciratefi(self):
        results = ciratefi.ciratefi(self.template, self.search, upsampling=10, cifi_thresh=self.cifi_thresh,
                                    rafi_thresh=self.rafi_thresh, tefi_thresh=self.tefi_thresh,
                                    use_percentile=self.use_percentile, alpha=self.alpha, radii=self.radii)

        self.assertEqual(len(results), 3)
        self.assertTrue((np.array(results[1], results[0]) < 1).all())

        results = ciratefi.ciratefi(self.offset_template, self.search, upsampling=self.upsampling,
                                    cifi_thresh=self.cifi_thresh, rafi_thresh=self.rafi_thresh,
                                    tefi_thresh=self.tefi_thresh,
                                    use_percentile=self.use_percentile, alpha=self.alpha, radii=self.radii)



    def tearDown(self):
        pass
