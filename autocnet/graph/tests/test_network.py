import os
import unittest

from autocnet.examples import get_path

import sys
sys.path.insert(0, os.path.abspath('..'))

from .. import network


class TestCandidateGraph(unittest.TestCase):
    
    def setUp(self):
        self.graph = network.CandidateGraph.from_adjacency(get_path('adjacency.json'))
    
    def test_add_image(self):
        self.graph.add_image("A")
        self.graph.add_image("PI")
        self.graph.add_image("OTHER")
        truth = ['AS15-M-0298_SML.png',
                 'AS15-M-0300_SML.png',
                 'AS15-M-0295_SML.png',
                 'AS15-M-0299_SML.png',
                 'AS15-M-0296_SML.png',
                 'AS15-M-0297_SML.png']
        truth += ['A', 'PI', 'OTHER']
        self.assertEqual(sorted(self.graph.nodes()), sorted(truth))

    def test_add_image_fail(self):
        import numpy as np
        import scipy.misc
        print(dir(scipy.misc))
        print(scipy.__version__)
        scipy.misc.bytescale(np.arange(100).reshape(10,10))

        self.assertRaises(TypeError, self.graph.add_image, [1, 2, 3])

    def test_adjacency_to_json(self):
        self.graph.adjacency_to_json('test_adjacency_to_json.json')
        self.assertTrue(os.path.exists('test_adjacency_to_json.json'))

    def tearDown(self):
        try:
            os.remove('test_adjacency_to_json.json')
        except FileExistsError:
            pass
