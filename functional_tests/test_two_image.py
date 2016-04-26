import os

import unittest

from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist
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

        cg.match_features(k=2)

        # Perform the symmetry check
        cg.symmetry_checks()
        # Perform the ratio check
        cg.ratio_checks(clean_keys = ['symmetry'])
        # Create fundamental matrix
        cg.compute_fundamental_matrices(clean_keys = ['symmetry', 'ratio'])


        for source, destination, edge in cg.edges_iter(data=True):

            # Perform the symmetry check
            self.assertIn(edge.masks['symmetry'].sum(), range(400, 600))
            # Perform the ratio test
            self.assertIn(edge.masks['ratio'].sum(), range(30, 100))

            # Range needs to be set
            self.assertIn(edge.masks['fundamental'].sum(), range(30, 50))


        # Step: Compute the homographies and apply RANSAC
        cg.compute_homographies(clean_keys=['symmetry', 'ratio'])

        # Step: Compute the overlap ratio and coverage ratio
        for s, d, edge in cg.edges_iter(data=True):
            edge.coverage_ratio(clean_keys=['symmetry', 'ratio'])

        # Step: Compute subpixel offsets for candidate points
        cg.subpixel_register(clean_keys=['ransac'])

        # Step:
        cg.suppress(clean_keys = 'ratio')

        # Step: And create a C object
        cnet = cg.to_cnet(clean_keys=['symmetry', 'ratio', 'ransac', 'subpixel'])

        # Step: Create a fromlist to go with the cnet and write it to a file
        filelist = cg.to_filelist()
        write_filelist(filelist, path="fromlis.lis")

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
            os.remove('TestTwoImageMatching.net')
            os.remove('fromlist.lis')
        except: pass
