import unittest

import numpy as np
import networkx as nx

from .. import markov_cluster

from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.io.network import load

class TestMarkovCluster(unittest.TestCase):

    def setUp(self):
        self.g = load(get_path('sixty_four_apollo.proj'))

    def test_mcl_from_network(self):
        self.g.compute_clusters(inflate_factor=15)
        self.assertIsInstance(self.g.clusters, dict)
        self.assertEqual(len(self.g.clusters), 14)

    def test_mcl_from_adj_matrix(self):
        arr = np.array(nx.adjacency_matrix(self.g).todense())
        flow, clusters = markov_cluster.mcl(arr)
        self.assertIsInstance(clusters, dict)
        self.assertEqual(len(clusters), 3)
