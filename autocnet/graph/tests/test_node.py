import os
import sys

import unittest
from unittest.mock import Mock, MagicMock
import warnings

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LinearRing


from autocnet.examples import get_path
from plio.io.io_gdal import GeoDataset

from .. import node

sys.path.insert(0, os.path.abspath('..'))


class TestNode(object):

    @pytest.fixture
    def node(self):
        img = get_path('AS15-M-0295_SML.png')
        return node.Node(image_name='AS15-M-0295_SML',
                              image_path=img)

    @pytest.fixture
    def geo_node(self):
        img = get_path('AS15-M-0297_crop.cub')
        return node.Node(image_name='AS15-M-0297_crop.cub', image_path=img)

    def test_get_handle(self, node):
        assert isinstance(node.geodata, GeoDataset)

    def test_get_byte_array(self, node):
        image = node.get_byte_array()
        assert (1012, 1012) == image.shape
        assert np.uint8 == image.dtype

    def test_equalities(self,node_a, node_b):
        assert node_a < node_b
        assert node_a <= node_b
        assert not (node_a > node_b)
        assert not (node_a >= node_b)
        assert not (node_a == node_b)

        node_a.random_attr = np.arange(10)
        node_b.random_attr = np.arange(10)

        assert not (node_a == node_b)

    def test_get_array(self, node):
        image = node.get_array()
        assert (1012, 1012) == image.shape
        assert np.float32 == image.dtype

    def test_extract_features(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_parameters={'nfeatures': 10})
        assert len(node.get_keypoints()) ==  10
        assert len(node.descriptors) == 10
        assert 10 == node.nkeypoints

    def test_extract_downsampled_features(self, node):
        # Trust that the
        img = np.random.random(size=(1000,1000))
        geodata = Mock(spec=GeoDataset)
        geodata.raster_size = img.shape
        geodata.read_array = MagicMock(return_value=img)
        node.extract_features_with_downsampling(5,
                                                extractor_parameters={'nfeatures':10})

        assert len(node.keypoints) in range(8,12)
        assert node.keypoints['x'].max() > 500

    def test_extract_tiled_features(self, node):
        tilesize = 500
        node.extract_features_with_tiling(tilesize=tilesize, overlap=50,
                                          extractor_parameters={'nfeatures':10})

        kps = node.keypoints
        assert kps['x'].min() < tilesize
        assert kps['y'].min() < tilesize
        assert len(kps) == pytest.approx(90, 3)

        with pytest.raises(ValueError) as e_info:
            node.extract_features_with_tiling(tilesize=10, overlap=20)

    def test_masks(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_parameters={'nfeatures': 5})
        assert isinstance(node.masks, pd.DataFrame)
        # Create an artificial mask
        node.masks['foo'] =  np.array([0, 0, 1, 1, 1], dtype=np.bool)
        assert node.masks['foo'].sum() == 3

    def test_convex_hull_ratio_fail(self):
        # Convex hull computation is checked lower in the hull computation
        #self.assertRaises(AttributeError, node.coverage_ratio)
        pass

    def test_coverage(self):
        image = self.node.get_array()
        self.node.extract_features(image, method='sift', extractor_parameters={'nfeatures': 10})

        coverage_percn = self.node.coverage()

        self.assertAlmostEqual(coverage_percn, 38.06139557)

    def test_isis_serial(self, node):
        serial = node.isis_serial
        assert None == serial

    def test_save_load(self, node, tmpdir):
        # Test that without keypoints this warns
        with pytest.warns(UserWarning) as warn:
            node.save_features(tmpdir.join('noattr.npy'))
        assert len(warn) == 1

        basename = tmpdir.dirname

        # With keypoints to npy
        reference = pd.DataFrame(np.arange(10).reshape(5,2), columns=['x', 'y'])
        node.keypoints = reference
        tmpdir.join('kps')
        node.save_features(os.path.join(basename, 'kps'))
        node.keypoints = None
        node.load_features(os.path.join(basename, 'kps_None.npz'))
        assert node.keypoints.equals(reference)

    def test_coverage(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_method='sift', extractor_parameters={'nfeatures': 10})
        coverage_percn = node.coverage()
        assert coverage_percn == pytest.approx(0.3806139557, 2)

    def test_clean(self, node):
        with pytest.raises(AttributeError):
            node._clean([])
        node.keypoints = pd.DataFrame(np.arange(5))
        node.masks = pd.DataFrame(np.array([[True, True, True, False, False],
                                   [True, False, True, True, False]]).T,
                                   columns=['a', 'b'])
        matches, mask = node._clean(clean_keys=['a'])
        assert mask.equals(pd.Series([True, True, True, False, False]))

    def test_get_keypoints(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_parameters={'nfeatures':5})
        kps = node.get_keypoints(index=[1,3])
        assert len(kps) == 2
        assert 1 in kps.index and 3 in kps.index

    def test_get_keypoint_coordinates(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_parameters={'nfeatures':5})
        kpc = node.get_keypoint_coordinates()
        assert 'x' in kpc.columns
        assert 'y' in kpc.columns
        kpc = node.get_keypoint_coordinates(index=[2,4])
        assert len(kpc) == 2
        kpc = node.get_keypoint_coordinates(homogeneous=True)
        assert (kpc.homogeneous == 1).all()

    def test_get_raw_keypoint_coordinates(self, node):
        image = node.get_array()
        node.extract_features(image, extractor_parameters={'nfeatures':5})
        kpc = node.get_raw_keypoint_coordinates()
        assert isinstance(kpc, np.ndarray)
        assert kpc.shape == (5,2)

        kpc = node.get_raw_keypoint_coordinates(-1)
        assert kpc.shape == (2,)


    def test_footprint(self, geo_node, node_a):
        # Esnure that a shapely compliant poly is being returned
        print(geo_node.footprint)
        assert isinstance(geo_node.footprint, LinearRing)
        assert node_a.footprint == None
