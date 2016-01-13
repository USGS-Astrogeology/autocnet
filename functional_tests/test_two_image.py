import os
import unittest

from scipy.misc import bytescale

from autocnet.control import C
from autocnet.examples import get_path
from autocnet.fileio.io_gdal import GeoDataset
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.graph.network import CandidateGraph


class TestTwoImageMatching(unittest.TestCase):
    """
    Feature: As a user
        I wish to automatically match two images to
        Generate an ISIS control network

        Scenario: Match two images
            Given a manually specified adjacency structure named two_image_adjacency.json
            When read create an adjacency graph
            Then extract image data and attribute nodes
            And find features and descriptors
            Then tag these to the graph nodes
            Then apply a FLANN matcher
            And create a C object
            Then output a control network
    """

    def test_two_image(self):
        # Step: Create an adjacency graph
        adjacency = get_path('two_image_adjacency.json')
        basepath = os.path.dirname(adjacency)
        cg = CandidateGraph.from_adjacency(adjacency)
        self.assertEqual(2, cg.number_of_nodes())
        self.assertEqual(2, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        for node, attributes in cg.nodes_iter(data=True):
            dataset = GeoDataset(os.path.join(basepath, node))
            attributes['handle'] = dataset
            attributes['image'] = bytescale(dataset.read_array())

        # Step: Then find features and descriptors

        # Step: And tag these to the graph nodes
        for node, attributes in cg.nodes_iter(data=True):
            attributes['features'] = 'a'  # Will be our feature/descriptor data structure
            attributes['descriptors'] = 'b'

        # Step: Then apply a FLANN matcher

        # Step: And create a C object
        cnet = C()
        # Step: Output a control network
        to_isis('TestTwoImageMatching.net', cnet, mode='wb',
                networkid='TestTwoImageMatching', targetname='Moon')

        self.assertTrue(False)
