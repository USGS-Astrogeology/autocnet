import unittest
from .. import naive_template
from numpy import array
from numpy import uint8


class TestNaiveTemplate(unittest.TestCase):

    def setUp(self):
        # Center is (5, 6)
        self._test_image = array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                                  (0, 0, 0, 0, 0, 0, 0, 1, 0),
                                  (1, 1, 1, 0, 0, 0, 0, 1, 0),
                                  (0, 1, 0, 0, 0, 0, 0, 0, 0),
                                  (0, 1, 0, 0, 0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                  (0, 0, 0, 0, 0, 0, 1, 1, 1),
                                  (0, 1, 1, 1, 0, 0, 1, 0, 1),
                                  (0, 1, 0, 1, 0, 0, 1, 0, 1),
                                  (0, 1, 1, 1, 0, 0, 1, 0, 1),
                                  (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=uint8)

        # Should yield (-3, 3) offset from image center
        self._t_shape = array(((1, 1, 1),
                               (0, 1, 0),
                               (0, 1, 0)), dtype=uint8)

        # Should be (3, -4)
        self._rect_shape = array(((1, 1, 1),
                                  (1, 0, 1),
                                  (1, 0, 1),
                                  (1, 0, 1),
                                  (1, 1, 1)), dtype=uint8)

        # Should be (-2, -4)
        self._square_shape = array(((1, 1, 1),
                                    (1, 0, 1),
                                    (1, 1, 1)), dtype=uint8)

        # Should be (3, 5)
        self._vertical_line = array(((0, 1, 0),
                                     (0, 1, 0),
                                     (0, 1, 0)), dtype=uint8)

    def test_t_shape(self):
        result_x, result_y, result_strength = naive_template.pattern_match(self._t_shape,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, -3)
        self.assertEqual(result_y, 3)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_rect_shape(self):
        result_x, result_y, result_strength = naive_template.pattern_match(self._rect_shape,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 3)
        self.assertEqual(result_y, -4)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_square_shape(self):
        result_x, result_y, result_strength = naive_template.pattern_match(self._square_shape,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, -2)
        self.assertEqual(result_y, -4)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_line_shape(self):
        result_x, result_y, result_strength = naive_template.pattern_match(self._vertical_line,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 3)
        self.assertEqual(result_y, 5)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def tearDown(self):
        pass

