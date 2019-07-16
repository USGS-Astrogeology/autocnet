import pytest
from unittest.mock import MagicMock, patch
from shapely.geometry import Polygon
from autocnet.spatial.overlap import place_points_in_overlap, place_points_in_overlaps
from autocnet.graph.node import Node
import csmapi


@patch('autocnet.cg.cg.distribute_points_in_geom', return_value=[(0, 0), (5, 5), (10, 10)])
def test_place_points_in_overlap(point_distributer):
    # Mock setup
    first_node = MagicMock()
    first_node.camera = MagicMock()
    first_node.camera.groundToImage.return_value = csmapi.ImageCoord(1.0, 0.0)
    first_node.isis_serial = '1'
    first_node.__getitem__.return_value = 1
    second_node = MagicMock()
    second_node.camera = MagicMock()
    second_node.camera.groundToImage.return_value = csmapi.ImageCoord(1.0, 1.0)
    second_node.isis_serial = '2'
    second_node.__getitem__.return_value = 2
    third_node = MagicMock()
    third_node.camera = MagicMock()
    third_node.camera.groundToImage.return_value = csmapi.ImageCoord(0.0, 1.0)
    third_node.isis_serial = '3'
    third_node.__getitem__.return_value = 3
    fourth_node = MagicMock()
    fourth_node.camera = MagicMock()
    fourth_node.camera.groundToImage.return_value = csmapi.ImageCoord(0.0, 0.0)
    fourth_node.isis_serial = '4'
    fourth_node.__getitem__.return_value = 4

    # Actual function being tested
    points = place_points_in_overlap([first_node, second_node, third_node, fourth_node],
                                      Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]))

    # Check the function output
    assert len(points) == 3
    for point in points:
        measure_ids = [measure.imageid for measure in point.measures]
        measure_serials = [measure.serial for measure in point.measures]
        assert measure_ids == [1, 2, 3, 4]
        assert measure_serials == ['1', '2', '3', '4']

    # Check the mocks
    point_distributer.assert_called_with(Polygon([(0, 0), (0, 10), (10, 10), (10, 0)]))
    first_node.camera.groundToImage.assert_called()
    second_node.camera.groundToImage.assert_called()
    third_node.camera.groundToImage.assert_called()
    fourth_node.camera.groundToImage.assert_called()

class MockOverlap():
    intersections = [0,1]
    geom = Polygon([(0,0),(0,5),(5,5),(5,0),(0,0)])



@patch('autocnet.io.db.model.Overlay.overlapping_larger_than', return_value=[MockOverlap()]*3)
@patch('autocnet.io.db.model.Points.bulkadd')
@pytest.mark.parametrize("distributekwargs",[
    ({'distribute_points_kwargs':{'ewpts_func':lambda:True}})
])
def test_place_points_in_overlaps(overlapper, adder, distributekwargs):
    nodes = [{"id": 0, "data": Node()}, {"id": 1, "data": Node()}]
    with patch('autocnet.spatial.overlap.place_points_in_overlap', return_value=[1,2,3]) as ppio:
        place_points_in_overlaps(nodes,distribute_points_kwargs=distributekwargs)
        ppio.assert_called_with([Node(), Node()],
                                Polygon([(0,0),(0,5),(5,5),(5,0),(0,0)]),
                                distribute_points_kwargs=distributekwargs,
                                cam_type='csm')
