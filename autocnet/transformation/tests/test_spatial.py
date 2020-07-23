from unittest import mock
import pytest
import math

from autocnet.transformation import spatial

def test_oc2og():
    lon = 0
    lat = 20
    lon_og, lat_og = spatial.oc2og(lon, lat, 3396190, 3376200)

    # calculate lat conversion by hand
    dlat = math.radians(lat)
    dlat = math.atan(((3396190 / 3376200)**2) * (math.tan(dlat)))
    dlat = math.degrees(dlat)

    assert math.isclose(lat_og, dlat)

def test_og2oc():
    lon = 0
    lat = 20
    lon_oc, lat_oc = spatial.og2oc(lon, lat, 3396190, 3376200)

    dlat = math.radians(lat)
    dlat = math.atan((math.tan(dlat) / ((3396190 / 3376200)**2)))
    dlat = math.degrees(dlat)

    assert math.isclose(lat_oc, dlat)


def test_reproject():
    with mock.patch('pyproj.transform', return_value=[1,1,1]) as mock_pyproj:
        res = spatial.reproject([1,1,1], 10, 10, 'geocent', 'latlon')
        mock_pyproj.assert_called_once()
        assert res == (1,1,1)
