import pyproj
import numpy as np

def og2oc(lon, lat, semi_major, semi_minor):
    """
    Converts planetographic latitude to planetocentric latitude using pyproj pipeline.

    Parameters
    ----------
    lon : float
          longitude 0 to 360 domain (in degrees)

    lat : float
          planetographic latitude (in degrees)

    semi_major : float
                 Radius from the center of the body to the equator

    semi_minor : float
                 Radius from the center of the body to the pole

    Returns
    -------
    lon: float
         longitude (in degrees)

    lat: float
         planetocentric latitude (in degrees)
    """
    lon_og = np.radians(lon)
    lat_og = np.radians(lat)

    proj_str = f"""
    +proj=pipeline
    +step +proj=geoc +a={semi_major} +b={semi_minor} +lon_wrap=180 +xy_in=rad +xy_out=rad
    """
    og2oc = pyproj.transformer.Transformer.from_pipeline(proj_str)
    lon_oc, lat_oc = og2oc.transform(lon_og, lat_og, errcheck=True, radians=True)
    return np.degrees(lon_oc), np.degrees(lat_oc)

def oc2og(lon, lat, semi_major, semi_minor):
    """
    Converts planetocentric latitude to planetographic latitude using pyproj pipeline.

    Parameters
    ----------
    lon : float
          longitude 0 to 360 domain (in degrees)

    lat : float
          planetocentric latitude (in degrees)

    semi_major : float
                 Radius from the center of the body to the equator

    semi_minor : float
                 Radius from the center of the body to the pole

    Returns
    -------
    lon : float
          longitude (in degrees)

    lat : float
          planetographic latitude (in degrees)
    """

    lon_oc = np.radians(lon)
    lat_oc = np.radians(lat)

    proj_str = f"""
    +proj=pipeline
    +step +proj=geoc +a={semi_major} +b={semi_minor} +lon_wrap=180 +inv +xy_in=rad +xy_out=rad
    """
    oc2og = pyproj.transformer.Transformer.from_pipeline(proj_str)
    lon_og, lat_og = oc2og.transform(lon_oc, lat_oc, errcheck=True, radians=True)

    return np.degrees(lon_og), np.degrees(lat_og)

def reproject(record, semi_major, semi_minor, source_proj, dest_proj, **kwargs):
    """
    Thin wrapper around PyProj's Transform() function to transform 1 or more three-dimensional
    point from one coordinate system to another. If converting between Cartesian
    body-centered body-fixed (BCBF) coordinates and Longitude/Latitude/Altitude coordinates,
    the values input for semi-major and semi-minor axes determine whether latitudes are
    planetographic or planetocentric and determine the shape of the datum for altitudes.
    If semi_major == semi_minor, then latitudes are interpreted/created as planetocentric
    and altitudes are interpreted/created as referenced to a spherical datum.
    If semi_major != semi_minor, then latitudes are interpreted/created as planetographic
    and altitudes are interpreted/created as referenced to an ellipsoidal datum.

    Parameters
    ----------
    record : object
             Pandas series object

    semi_major : float
                 Radius from the center of the body to the equater

    semi_minor : float
                 Radius from the pole to the center of mass

    source_proj : str
                         Pyproj string that defines a projection space ie. 'geocent'

    dest_proj : str
                      Pyproj string that defines a project space ie. 'latlon'

    Returns
    -------
    : list
      Transformed coordinates as y, x, z

    """
    source_pyproj = pyproj.Proj(proj=source_proj, a=semi_major, b=semi_minor, lon_wrap=180)
    dest_pyproj = pyproj.Proj(proj=dest_proj, a=semi_major, b=semi_minor, lon_wrap=180)

    y, x, z = pyproj.transform(source_pyproj, dest_pyproj, record[0], record[1], record[2], **kwargs)
    return y, x, z
