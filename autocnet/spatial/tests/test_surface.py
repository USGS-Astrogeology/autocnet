import unittest

from unittest import mock

from .. import surface

class TestEllipsoidDem(unittest.TestCase):

    def test_height(self):
        test_dem = surface.EllipsoidDem(3396190, 3376200)
        self.assertEqual(test_dem.get_height(0, 0), 0)
        self.assertEqual(test_dem.get_height(0, 180), 0)
        self.assertEqual(test_dem.get_height(90, 100), 0)

    def test_radius(self):
        test_dem = surface.EllipsoidDem(3396190, 3376200)
        self.assertEqual(test_dem.get_radius(0, 0), 3396190)
        self.assertEqual(test_dem.get_radius(0, 180), 3396190)
        self.assertEqual(test_dem.get_radius(90, 300), 3376200)

    def tearDown(self):
        pass


class TestGdalDem(unittest.TestCase):

    def test_height(self):
        with mock.patch('autocnet.spatial.surface.GeoDataset') as mockDataset:
            mockInstance = mockDataset.return_value
            mockInstance.latlon_to_pixel.return_value = (1,2)
            mockInstance.read_array.return_value = [[100]]
            test_dem = surface.GdalDem('TestDem.cub', 3396190, 3376200)
            self.assertEqual(test_dem.get_height(0, 0), 100)
            self.assertEqual(test_dem.get_height(0, 180), 100)
            self.assertEqual(test_dem.get_height(90, 300), 100)

    def test_height_from_radius(self):
        with mock.patch('autocnet.spatial.surface.GeoDataset') as mockDataset:
            mockInstance = mockDataset.return_value
            mockInstance.latlon_to_pixel.return_value = (1,2)
            mockInstance.read_array.return_value = [[3396190]]
            test_dem = surface.GdalDem('TestDem.cub', 3396190, 3376200, 'radius')
            self.assertEqual(test_dem.get_height(0, 0), 0)
            self.assertEqual(test_dem.get_height(0, 180), 0)
            self.assertEqual(test_dem.get_height(90, 300), 19990)

    def test_radius(self):
        with mock.patch('autocnet.spatial.surface.GeoDataset') as mockDataset:
            mockInstance = mockDataset.return_value
            mockInstance.latlon_to_pixel.return_value = (1,2)
            mockInstance.read_array.return_value = [[3396190]]
            test_dem = surface.GdalDem('TestDem.cub', 3396190, 3376200, 'radius')
            self.assertEqual(test_dem.get_radius(0, 0), 3396190)
            self.assertEqual(test_dem.get_radius(0, 180), 3396190)
            self.assertEqual(test_dem.get_radius(90, 300), 3396190)

    def test_radius_from_height(self):
        with mock.patch('autocnet.spatial.surface.GeoDataset') as mockDataset:
            mockInstance = mockDataset.return_value
            mockInstance.latlon_to_pixel.return_value = (1,2)
            mockInstance.read_array.return_value = [[100]]
            test_dem = surface.GdalDem('TestDem.cub', 3396190, 3376200)
            self.assertEqual(test_dem.get_radius(0, 0), 3396290)
            self.assertEqual(test_dem.get_radius(0, 180), 3396290)
            self.assertEqual(test_dem.get_radius(90, 300), 3376300)

    def tearDown(self):
        pass
