import os

import unittest

from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist
from autocnet.graph.network import CandidateGraph
from autocnet.matcher.matcher import FlannMatcher


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
        basepath = get_path('Apollo15')
        cg = CandidateGraph.from_adjacency(adjacency, basepath=basepath)
        self.assertEqual(2, cg.number_of_nodes())
        self.assertEqual(1, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        cg.extract_features(method='sift', extractor_parameters={"nfeatures":500})
        for i, node in cg.nodes_iter(data=True):
            self.assertIn(node.nkeypoints, range(490, 511))

        #Step: Compute the coverage ratios
        truth_ratios = [0.95351579,
                        0.93595664]
        for i, node in cg.nodes_iter(data=True):
            ratio = node.coverage_ratio()
            self.assertIn(round(ratio,8), truth_ratios)
        # Step: apply Adaptive non-maximal suppression
        for i, node in cg.nodes_iter(data=True):
            pass
            #node.anms()
            #self.assertNotEqual(node.nkeypoints, sum(node._mask_arrays['anms']))

        # Step: Then apply a FLANN matcher
        fl = FlannMatcher()
        for i, node, in cg.nodes_iter(data=True):
            fl.add(node.descriptors, key=i)
        fl.train()

        for i, node in cg.nodes_iter(data=True):
            descriptors = node.descriptors
            matches = fl.query(descriptors, i, k=5)
            cg.add_matches(matches)

        for source, destination, edge in cg.edges_iter(data=True):
            # Perform the symmetry check
            edge.symmetry_check()
            self.assertIn(edge._mask_arrays['symmetry'].sum(), range(430, 461))

            # Perform the ratio test
            edge.ratio_check(ratio=0.8)
            self.assertIn(edge._mask_arrays['ratio'].sum(), range(250, 350))

        # Step: Compute the homographies and apply RANSAC
        cg.compute_homographies(clean_keys=['symmetry', 'ratio'])

        # Step: Compute the overlap ratio and coverage ratio
        for s, d, edge in cg.edges_iter(data=True):
            ratio = edge.coverage_ratio(clean_keys=['symmetry', 'ratio'])

        # Step: Compute subpixel offsets for candidate points
        cg.compute_subpixel_offsets(clean_keys=['ransac'])

        # Step: And create a C object
        cnet = cg.to_cnet(clean_keys=['symmetry', 'ratio', 'ransac', 'subpixel'])

        # Step: Create a fromlist to go with the cnet and write it to a file
        filelist = cg.to_filelist()
        write_filelist(filelist)

        # Step update the serial numbers
        nid_to_serial = {}
        for i, node in cg.nodes_iter(data=True):
            nid_to_serial[i] = self.serial_numbers[node.image_name]

        cnet.replace({'nid': nid_to_serial}, inplace=True)
        # Step: Output a control network
        to_isis('TestTwoImageMatching.net', cnet, mode='wb',
                networkid='TestTwoImageMatching', targetname='Moon')

    def tearDown(self):
        try:
            os.path.remove('TestTwoImageMatching.net')
            os.path.remove('fromlist.lis')
        except: pass
