import os
import unittest

from autocnet.examples import get_path

import sys

from .. import isis_serial_numbers

sys.path.insert(0, os.path.abspath('..'))


class TestIsisSerials(unittest.TestCase):

    def test_generate_serial_number(self):
        label = get_path('Test_PVL.lbl')
        serial = isis_serial_numbers.generate_serial_number(label)
        self.assertEqual('APOLLO15/METRIC/1971-07-31T14:02:27.179', serial)
