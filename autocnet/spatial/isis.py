import pvl
from pysis import isis
from warnings import warn
from pysis.exceptions import ProcessError
from numbers import Number
import numpy as np
import tempfile

def point_info(cube_path, x, y, point_type):
    """
    Use Isis's campt to get image/ground point info from an image

    Parameters
    ----------
    cube_path : str
                path to the input cube

    x : float
        point in the x direction. Either a sample or a longitude value
        depending on the point_type flag

    y : float
        point in the y direction. Either a line or a latitude value
        depending on the point_type flag

    point_type : str
                 Options: {"image", "ground"}
                 Pass "image" if  x,y are in image space (sample, line) or
                 "ground" if in ground space (longitude, lattiude)

    Returns
    -------
    : PvlObject
      Pvl object containing campt returns
    """
    point_type = point_type.lower()

    if point_type not in {"image", "ground"}:
        raise Exception(f'{point_type} is not a valid point type, valid types are ["image", "ground"]')


    if isinstance(x, Number) and isinstance(y, Number):
        x, y = [x], [y]

    with tempfile.NamedTemporaryFile("w+") as f:
        # ISIS wants points in a file, so write to a temp file
        if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            x,y = y,x
        elif point_type == "image":
            # convert to ISIS pixels
            x = np.add(x, .5)
            y = np.add(y, .5)

        f.write("\n".join(["{}, {}".format(xval,yval) for xval,yval in zip(x, y)]))
        f.flush()
        try:
            pvlres = isis.campt(from_=cube_path, coordlist=f.name ,usecoordlist=True, coordtype=point_type)
            for r in pvlres:
                # convert all pixels to PLIO pixels from ISIS
                r[1]["Sample"] -= .5
                r[1]["Line"] -= .5

        except ProcessError as e:
            warn(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
            return

        pvlres = pvl.loads(pvlres)

    return pvlres


def image_to_ground(cube_path, sample, line, lattype="PlanetocentricLatitude", lonttype="PositiveEast360Longitude"):
    """
    Use Isis's campt to convert a line sample point on an image to lat lon

    Returns
    -------
    lats : np.array, float
           1-D array of latitudes or single floating point latitude

    lons : np.array, float
           1-D array of longitudes or single floating point longitude

    """
    # campt always does x,y
    pvlres = point_info(cube_path, sample, line, "image")
    lats, lons = np.asarray([[r[1][lattype].value, r[1][lonttype].value] for r in pvlres]).T
    if len(lats) == 1 and len(lons) == 1:
        lats, lons = lats[0], lons[0]

    return lats, lons

def ground_to_image(cube_path, lon, lat):
    """
    Use Isis's campt to convert a lat lon point to line sample in
    an image

    Returns
    -------
    lines : np.array, float
            array of lines or single flaoting point line

    samples : np.array, float
              array of samples or single dloating point sample

    """

    pvlres = point_info(cube_path, lon, lat, "ground")
    lines, samples = np.asarray([[r[1]["Line"], r[1]["Sample"]] for r in pvlres]).T
    if len(lines) == 1 and len(samples) == 1:
        lines, samples = lines[0], samples[0]
    return lines, samples


