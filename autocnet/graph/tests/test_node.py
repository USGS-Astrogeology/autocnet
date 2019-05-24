import os
import sys

import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import warnings

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LinearRing


from autocnet.examples import get_path
from plio.io.io_gdal import GeoDataset

from autocnet.graph.node import Node

sys.path.insert(0, os.path.abspath('..'))


class TestNode(object):

    @pytest.fixture
    def node(self):
        return Node()

    @pytest.fixture
    def geo_node(self):
        img = get_path('AS15-M-0297_crop.cub')
        return Node(image_name='AS15-M-0297_crop.cub', image_path=img)

    def test_get_camera(self, node):
        assert node.camera == None
    
    def test_create(self):
        assert isinstance(Node.create(None, 1), Node)
        n = Node.create('foo', 1, basepath='/')
        assert isinstance(n, Node)
        assert n['image_path'] == '/foo'
        assert n['image_name'] == 'foo'

    def test_set_camera(self, node):
        node.camera = 'foo'
        assert node.camera == 'foo'

    def test_get_handle(self, geo_node):
        assert isinstance(geo_node.geodata, GeoDataset)

    def test_get_byte_array(self, node):
        dtype = np.float32
        return_value = np.arange(9, dtype=dtype).reshape(3,3)
        
        mock_geodata = Mock(spec=GeoDataset)
        mock_geodata.read_array = MagicMock(return_value=return_value)
        node._geodata = mock_geodata

        image = node.get_byte_array()
        assert return_value.shape == image.shape
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
        dtype = np.float32
        return_value = np.arange(9, dtype=dtype).reshape(3,3)
        
        mock_geodata = Mock(spec=GeoDataset)
        mock_geodata.read_array = MagicMock(return_value=return_value)
        node._geodata = mock_geodata

        image = node.get_array()
        assert return_value.shape == image.shape
        assert dtype == image.dtype

    def test_extract_features(self, node):
        desc = np.arange(9).reshape(3,3)
        kps = pd.DataFrame(desc)
        return_value = (kps, desc)
        image = desc
        with patch('autocnet.graph.node.Node._extract_features', return_value=return_value):
            node.extract_features(image, extractor_parameters={'nfeatures': 10})
            assert len(node.get_keypoints()) ==  3
            assert len(node.descriptors) == 3
            assert 3 == node.nkeypoints

    @pytest.mark.filterwarnings("ignore::UserWarning")  # For skimage user warning for API changes
    def test_extract_downsampled_features(self, geo_node):
        desc = np.arange(9).reshape(3,3)
        kps = pd.DataFrame(desc, columns=['x', 'y', 'z'])
        with patch('autocnet.graph.node.Node._extract_features', return_value=(kps, desc)) as ef:
            geo_node.extract_features_with_downsampling(downsample_amount=3)
            assert ef.call_count == 1
            assert geo_node.keypoints[['x', 'y']].equals(kps[['x', 'y']] * 3)


    def test_extract_tiled_features(self, geo_node):
        tilesize = 100
        desc = np.arange(9).reshape(3,3)
        kps = pd.DataFrame(desc, columns=['x', 'y', 'z'])
        with patch('autocnet.graph.node.Node._extract_features', return_value=(kps, desc)) as ef:
            geo_node.extract_features_with_tiling(tilesize=tilesize,overlap=5)
            assert ef.call_count == 36 # 6 slices in the x and 6 slices in the y

    def test_masks(self, node):
        assert isinstance(node.masks, pd.DataFrame)
        # Create an artificial mask
        node.masks['foo'] =  np.array([0, 0, 1, 1, 1], dtype=np.bool)
        assert node.masks['foo'].sum() == 3

    def test_convex_hull_ratio_fail(self):
        # Convex hull computation is checked lower in the hull computation
        #self.assertRaises(AttributeError, node.coverage_ratio)
        pass

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
        kps = pd.DataFrame(np.array([[0,0],[1,1], [1,2], [2,2]]), columns=['x', 'y'])
        extent = (2,2)
        mock_geodata = Mock(spec=GeoDataset)
        mock_geodata.raster_size = extent
        node._geodata = mock_geodata
        with patch('autocnet.graph.node.Node.keypoints', new_callable=PropertyMock, return_value=kps):
            assert node.coverage() == 0.25

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
        kps = pd.DataFrame(np.arange(9).reshape(3,3))
        with patch('autocnet.graph.node.Node.keypoints', new_callable=PropertyMock, return_value=kps):
            assert len(node.get_keypoints()) == 3
            assert len(node.get_keypoints(index=[1,2])) == 2
            assert len(node.get_keypoints(index=[0])) == 1

    def test_get_keypoint_coordinates(self, node):
        kps = pd.DataFrame(np.arange(9).reshape(3,3), columns=['x', 'y', 'z'])
        with patch('autocnet.graph.node.Node.keypoints', new_callable=PropertyMock, return_value=kps):
            kpc = node.get_keypoint_coordinates()

            assert 'x' in kpc.columns
            assert 'y' in kpc.columns
            kpc = node.get_keypoint_coordinates(index=[0,2])
            assert len(kpc) == 2
            kpc = node.get_keypoint_coordinates(homogeneous=True)
            assert (kpc['homogeneous'] == 1).all()

    def test_get_raw_keypoint_coordinates(self, node):
        kps = pd.DataFrame(np.arange(9).reshape(3,3), columns=['x', 'y', 'z'])
        with patch('autocnet.graph.node.Node.keypoints', new_callable=PropertyMock, return_value=kps):
            kpc = node.get_raw_keypoint_coordinates()
            assert isinstance(kpc, np.ndarray) 
            assert kpc.shape == (3,2)
            kpc = node.get_raw_keypoint_coordinates(-1)
            assert kpc.shape == (2,)
        
