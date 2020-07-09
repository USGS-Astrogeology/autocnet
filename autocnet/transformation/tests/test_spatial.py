from unittest import mock
import pytest

from autocnet.transformation import spatial

def test_oc2og():
    lon = 0
    lat = 20
    lon_og, lat_og = spatial.oc2og(lon, lat, 3396190, 3376200)
    assert lat_og == 20.218400434636393

def test_og2oc():
    lon = 0
    lat = 20
    lon_oc, lat_oc = spatial.oc2og(lon, lat, 3396190, 3376200)
    assert lat_oc == 19.78356596059272


def test_reproject():
    with mock.patch('pyproj.transform', return_value=[1,1,1]) as mock_pyproj:
        res = spatial.reproject([1,1,1], 10, 10, 'geocent', 'latlon')
        mock_pyproj.assert_called_once()
        assert res == (1,1,1)
