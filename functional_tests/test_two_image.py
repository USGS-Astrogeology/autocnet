import os

import unittest
import numpy as np

from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.graph.network import CandidateGraph
from autocnet.matcher.matcher import FlannMatcher
from autocnet.matcher import outlier_detector as od


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
            And apply a FLANN matcher
            Then create a C object from the graph matches
            Then output a control network
    """

    def setUp(self):
        self.serial_numbers = {'AS15-M-0295_SML.png': '1971-07-31T01:24:11.754',
                               'AS15-M-0296_SML.png': '1971-07-31T01:24:36.970',
                               'AS15-M-0297_SML.png': '1971-07-31T01:25:02.243',
                               'AS15-M-0298_SML.png': '1971-07-31T01:25:27.457',
                               'AS15-M-0299_SML.png': '1971-07-31T01:25:52.669',
                               'AS15-M-0300_SML.png': '1971-07-31T01:26:17.923'}

        for k, v in self.serial_numbers.items():
            self.serial_numbers[k] = 'APOLLO15/METRIC/{}'.format(v)

    def test_two_image(self):
        # Step: Create an adjacency graph
        adjacency = get_path('two_image_adjacency.json')
        cg = CandidateGraph.from_adjacency_file(adjacency)
        self.assertEqual(2, cg.number_of_nodes())
        self.assertEqual(1, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        cg.extract_features(500)
        for node, attributes in cg.nodes_iter(data=True):
            self.assertIn(len(attributes['keypoints']), range(490, 511))

        # Step: Then apply a FLANN matcher
        fl = FlannMatcher()
        for node, attributes in cg.nodes_iter(data=True):
            fl.add(attributes['descriptors'], key=node)
        fl.train()

        for node, attributes in cg.nodes_iter(data=True):
            descriptors = attributes['descriptors']
            matches = fl.query(descriptors, node, k=5)
            cg.add_matches(matches)

        for source, destination, attributes in cg.edges_iter(data=True):
            matches = attributes['matches']
            # Perform the symmetry check
            symmetry_mask = od.mirroring_test(matches)
            self.assertIn(symmetry_mask.sum(), range(430, 461))
            attributes['symmetry'] = symmetry_mask

            # Perform the ratio test
            ratio_mask = od.distance_ratio(matches, ratio=0.95)
            self.assertIn(ratio_mask.sum(), range(390, 451))
            attributes['ratio'] = ratio_mask

            mask = np.array(ratio_mask * symmetry_mask)
            self.assertIn(len(matches.loc[mask]), range(75,101))

        cg.compute_homographies(clean_keys=['symmetry', 'ratio'])

        #compute subpixel offsets for the entire graph
        offsets = cg.compute_subpixel_offsets()
        self.assertEqual(len(offsets), cg.number_of_edges())

        # Step: And create a C object
        cnet = cg.to_cnet(clean_keys=['symmetry', 'ratio', 'ransac'])
        # Step update the serial numbers
        nid_to_serial = {}
        for node, attributes in cg.nodes_iter(data=True):
            nid_to_serial[node] = self.serial_numbers[attributes['image_name']]

        cnet.replace({'nid': nid_to_serial}, inplace=True)

        # Step: Output a control network
        to_isis('TestTwoImageMatching.net', cnet, mode='wb',
                networkid='TestTwoImageMatching', targetname='Moon')

    def tearDown(self):
        try:
            os.path.remove('TestTwoImageMatching.net')
        except: pass
