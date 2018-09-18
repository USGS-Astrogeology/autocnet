import os
import sys
import unittest
import warnings

import cv2

from .. import cpu_matcher
from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph

sys.path.append(os.path.abspath('..'))


"""class TestMatcher(unittest.TestCase):

    def setUp(self):
        im1 = cv2.imread(get_path('AS15-M-0296_SML.png'))
        im2 = cv2.imread(get_path('AS15-M-0297_SML.png'))

        self.fd = {}

        sift = cv2.xfeatures2d.SIFT_create(10)

        self.fd['AS15-M-0296_SML.png'] = sift.detectAndCompute(im1, None)
        self.fd['AS15-M-0297_SML.png'] = sift.detectAndCompute(im2, None)

    def test_flann_match_k_eq_2(self):
        fmatcher = cpu_matcher.FlannMatcher()
        source_image = self.fd['AS15-M-0296_SML.png']
        fmatcher.add(source_image[1], 0)

        self.assertTrue(len(fmatcher.nid_lookup), 1)

        fmatcher.train()

        with warnings.catch_warnings(record=True) as w:
            fmatcher.query(self.fd['AS15-M-0296_SML.png'][1], 0, k=2)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)

    def test_cpu_match(self):
        # Build a graph
        adjacency = get_path('two_image_adjacency.json')
        basepath = get_path('Apollo15')
        cang = CandidateGraph.from_adjacency(adjacency, extractor_method='vlfeat', basepath=basepath)

        # Extract features
        cang.extract_features(extractor_parameters={'nfeatures': 700})

        # Make sure cpu matcher is used for test
        edges = list()
        from autocnet.matcher.cpu_matcher import match as match
        for s, d in cang.edges():
            cang[s][d]['data']._match = match
            edges.append(cang[s][d])

        # Assert none of the edges have masks yet
        for edge in edges:
            self.assertTrue(edge['data'].masks.empty)

        # Match & outlier detect
        cang.match()
        cang.symmetry_checks()

        # Grab the length of a matches df
        match_len = len(edges[0]['data'].matches.index)

        # Assert symmetry check is now in all edge masks
        for edge in edges:
            self.assertTrue('symmetry' in edge['data'].masks)

        # Assert matches have been populated
        for edge in edges:
            self.assertTrue(not edge['data'].matches.empty)

        # Re-match
        cang.match()

        # Assert that new matches have been added on to old ones
        self.assertEqual(len(edges[0]['data'].matches.index), match_len * 2)

        # Assert that the match cleared the masks df
        for edge in edges:
            self.assertTrue(edge['data'].masks.empty)


    def tearDown(self):
        pass"""
