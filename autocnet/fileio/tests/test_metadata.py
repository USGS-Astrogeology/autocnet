import unittest
from osgeo import osr

from .. import extract_metadata as em
osr.UseExceptions()


class TestSRSProjectionExtraction(unittest.TestCase):

    def setUp(self):
        self.wktsrs = 'PROJCS["Moon2000_Mercator180",GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",180.0],PARAMETER["Standard_Parallel_1",0.0],UNIT["Meter",1.0]]'
        self.wktsrs = 'PROJCS["Mercator",GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",180.0],PARAMETER["Standard_Parallel_1",0.0],UNIT["Meter",1.0]]'
        self.srs = em.extract_projstring(self.wktsrs)

    def test_generate_srs(self):
        self.srs = em.extract_projstring(self.wktsrs)

    def test_false_easting(self):
        e = em.get_false_easting(self.srs)
        self.assertEqual(e, 0.0)

    def test_false_northing(self):
        n = em.get_false_northing(self.srs)
        self.assertEqual(n, 0.0)

    def test_projection_name(self):
        name = em.get_projection_name(self.srs)
        self.assertEqual(name, 'Mercator_1SP')

    def test_axes_extract(self):
        smajor, sminor, invflattening = em.get_spheroid(self.srs)
        self.assertEqual(smajor, 1737400.0)
        self.assertEqual(sminor, 1737400.0)
        self.assertEqual(invflattening, 0.0)

    def test_get_standard_parallels(self):
        parallels = em.get_standard_parallels(self.srs)
        self.assertEqual(parallels[0], 0.0)
        self.assertEqual(parallels[1], 0.0)

    def test_get_central_meridian(self):
        clon = em.get_central_meridian(self.srs)
        self.assertEqual(clon, 180.0)

    def test_export_to_proj4(self):
        """
        Check that proj4 is not supporting Moon2000_Mercator
        """
        proj4 = self.srs.ExportToProj4()
        self.assertEqual(
            proj4,
            '+proj=merc +lon_0=180 +k=1 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs ')

    def test_scale_factor(self):
        k = em.get_scale_factor(self.srs)
        self.assertEqual(k, 1.0)

    def test_latitude_of_origin(self):
        lo = em.get_latitude_of_origin(self.srs)
        self.assertEqual(lo, 0.0)
