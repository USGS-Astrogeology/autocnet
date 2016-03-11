import unittest

from .. import edge


class TestEdge(unittest.TestCase):

    def setUp(self):
        self.edge = edge.Edge(source=0, destination=1)

    def test_properties(self):
        pass
