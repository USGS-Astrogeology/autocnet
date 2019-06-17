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

    Returns
    -------
    : Line
      The line

    : Sample
      The sample

    """
    if isinstance(x, Number) and isinstance(y, Number):
        x, y = [x], [y]

    with tempfile.NamedTemporaryFile("w+") as f:
        # ISIS wants points in a file, so write to a temp file
        f.write("\n".join(["{}, {}".format(xval,yval) for xval,yval in zip(x, y)]))
        f.flush()
        try:
            pvlres = isis.campt(from_=cube_path, coordlist=f.name ,usecoordlist=True, coordtype=point_type)
        except ProcessError as e:
            warn(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
            return

        pvlres = pvl.loads(pvlres)

    return pvlres


def image_to_ground(cube_path, line, sample, lattype="PlanetocentricLatitude", lonttype="PositiveEast360Longitude"):
    """
    Use Isis's campt to convert a lat lon point to line sample in
    an image

    Returns
    -------
    : Line
      The line

    : Sample
      The sample

    """
    # campt always does x,y
    pvlres = point_info(cube_path, sample, line, "image")
    lats, lons = np.asarray([[r[1][lattype].value, r[1][lonttype].value] for r in pvlres]).T
    if len(lats) == 1 and len(lons) == 1:
        lats, lons = lats[0], lons[0]

    return lats, lons

def ground_to_image(cube_path, lat, lon):
    """
    Use Isis's campt to convert a lat lon point to line sample in
    an image

    Returns
    -------
    : Line
      The line

    : Sample
      The sample

    """

    pvlres = point_info(cube_path, lat, lon, "ground")
    lines, samples = np.asarray([[r[1]["Line"], r[1]["Sample"]] for r in pvlres]).T
    if len(lines) == 1 and len(samples) == 1:
        lines, samples = lines[0], samples[0]
    return lines, samples


