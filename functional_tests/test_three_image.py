import unittest

from autocnet.examples import get_path
from autocnet.fileio.io_controlnetwork import to_isis
from autocnet.fileio.io_controlnetwork import write_filelist
from autocnet.graph.network import CandidateGraph


class TestThreeImageMatching(unittest.TestCase):
    """
    Feature: As a user
        I wish to automatically match three images to
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
        self.serial_numbers = {
            'AS15-M-0295_SML.png': '1971-07-31T01:24:11.754',
            'AS15-M-0296_SML.png': '1971-07-31T01:24:36.970',
            'AS15-M-0297_SML.png': '1971-07-31T01:25:02.243',
            'AS15-M-0298_SML.png': '1971-07-31T01:25:27.457',
            'AS15-M-0299_SML.png': '1971-07-31T01:25:52.669',
            'AS15-M-0300_SML.png': '1971-07-31T01:26:17.923'}

        for k, v in self.serial_numbers.items():
            self.serial_numbers[k] = 'APOLLO15/METRIC/{}'.format(v)

    def test_three_image(self):
        # Step: Create an adjacency graph
        adjacency = get_path('three_image_adjacency.json')
        basepath = get_path('Apollo15')
        cg = CandidateGraph.from_adjacency(adjacency, basepath)
        self.assertEqual(3, cg.number_of_nodes())
        self.assertEqual(3, cg.number_of_edges())

        # Step: Extract image data and attribute nodes
        cg.extract_features(extractor_parameters={'nfeatures': 500})
        for i, node, in cg.nodes_iter(data=True):
            self.assertIn(node.nkeypoints, range(490, 511))

        cg.match_features(k=5)

        for source, destination, edge in cg.edges_iter(data=True):
            edge.symmetry_check()
            edge.ratio_check(clean_keys=['symmetry'], ratio=0.99)
        cg.compute_homographies(clean_keys=['symmetry', 'ratio'])

        # Step: And create a C object
        cnet = cg.to_cnet(clean_keys=['symmetry', 'ratio', 'ransac'])

        # Step: Create a fromlist to go with the cnet and write it to a file
        filelist = cg.to_filelist()
        write_filelist(filelist, 'TestThreeImageMatching_fromlist.lis')

        # Step update the serial numbers
        nid_to_serial = {}
        for i, node in cg.nodes_iter(data=True):
            nid_to_serial[i] = self.serial_numbers[node.image_name]

        cnet.replace({'nid': nid_to_serial}, inplace=True)
        # Step: Output a control network
        to_isis('TestThreeImageMatching.net', cnet, mode='wb',
                networkid='TestThreeImageMatching', targetname='Moon')

    def tearDown(self):
        try:
            os.path.remove('TestThreeImageMatching.net')
            os.path.remove('TestThreeImageMatching_fromlist.lis')
        except:
            pass
