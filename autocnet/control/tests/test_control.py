import os
import sys
from time import gmtime, strftime
import unittest

import numpy as np
# import pandas as pd

from autocnet.control import control

sys.path.insert(0, os.path.abspath('..'))


class TestC(unittest.TestCase):

    def setUp(self):
        x = list(range(10))
        y = list(range(10))
        pid = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        nid = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

        data = np.array([x, y, pid, nid]).T

        self.C = control.C(data, columns=['x', 'y', 'pid', 'nid'])

    def test_n_point(self):
        self.assertEqual(self.C.n, 4)

    def test_n_measures(self):
        self.assertEqual(self.C.m, 10)

    def test_modified_date(self):
        self.assertEqual(self.C.modifieddate, 'Not modified')

    def test_creation_date(self):
        self.assertEqual(
            self.C.creationdate, strftime(
                "%Y-%m-%d %H:%M:%S", gmtime()))
