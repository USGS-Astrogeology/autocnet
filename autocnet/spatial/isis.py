import pvl
from pysis import isis
from warnings import warn
from pysis.exceptions import ProcessError
from numbers import Number
import numpy as np
import tempfile


def point_info(cube_path, x, y, point_type, allow_outside=False):
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

    if point_type == "image":
        # convert to ISIS pixels
        x = np.add(x, .5)
        y = np.add(y, .5)

    if pvl.load(cube_path).get("IsisCube").get("Mapping"):
      pvlres = []
      # We have a projected image
      for x,y in zip(x,y):
        try:
          if point_type.lower() == "ground":
            pvlres.append(isis.mappt(from_=cube_path, longitude=x, latitude=y, allowoutside=allow_outside, coordsys="UNIVERSAL", type_=point_type))
          elif point_type.lower() == "image":
            pvlres.append(isis.mappt(from_=cube_path, sample=x, line=y, allowoutside=allow_outside, type_=point_type))
        except ProcessError as e:
          print(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
          return
      dictres = [dict(pvl.loads(res)["Results"]) for res  in pvlres]
      if len(dictres) == 1:
        pvlres = dictres[0]

    else:
      with tempfile.NamedTemporaryFile("w+") as f:
         # ISIS's campt wants points in a file, so write to a temp file
         if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            x,y = y,x


         f.write("\n".join(["{}, {}".format(xval,yval) for xval,yval in zip(x, y)]))
         f.flush()
         try:
            pvlres = isis.campt(from_=cube_path, coordlist=f.name, allowoutside=allow_outside, usecoordlist=True, coordtype=point_type)
         except ProcessError as e:
            warn(f"CAMPT call failed, image: {cube_path}\n{e.stderr}")
            return

         pvlres = pvl.loads(pvlres)
         dictres = []
         if len(x) > 1 and len(y) > 1:
            for r in pvlres:
                if r['GroundPoint']['Error'] is not None:
                    raise ProcessError(returncode=1, cmd=['pysis.campt()'], stdout=r, stderr=r['GroundPoint']['Error'])
                    return
                else:
                    # convert all pixels to PLIO pixels from ISIS
                    r[1]["Sample"] -= .5
                    r[1]["Line"] -= .5
                    dictres.append(dict(r[1]))
         else:
            if pvlres['GroundPoint']['Error'] is not None:
                raise ProcessError(returncode=1, cmd=['pysis.campt()'], stdout=pvlres, stderr=pvlres['GroundPoint']['Error'])
                return
            else:
                pvlres["GroundPoint"]["Sample"] -= .5
                pvlres["GroundPoint"]["Line"] -= .5
                dictres = dict(pvlres["GroundPoint"])
    return dictres


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
    try:
        res = point_info(cube_path, sample, line, "image")
    except ProcessError as e:
        raise ProcessError(returncode=e.returncode, cmd=e.cmd, stdout=e.stdout, stderr=e.stderr)

    try:
        if isinstance(res, list):
            lats, lons = np.asarray([[r[lattype].value, r[lonttype].value] for r in res]).T
        else:
            lats, lons = res[lattype].value, res[lonttype].value
    except Exception as e:
        if isinstance(res, list):
            lats, lons = np.asarray([[r[lattype], r[lonttype]] for r in res]).T
        else:
            lats, lons = res[lattype], res[lonttype]
    return lats, lons


def ground_to_image(cube_path, lon, lat):
    """
    Use Isis's campt to convert a lat lon point to line sample in
    an image

    Returns
    -------
    lines : np.array, float
            array of lines or single floating point line

    samples : np.array, float
              array of samples or single dloating point sample

    """
    try:
        res = point_info(cube_path, lon, lat, "ground")
    except ProcessError as e:
        raise ProcessError(returncode=e.returncode, cmd=e.cmd, stdout=e.stdout, stderr=e.stderr)

    try:
        if isinstance(res, list):
            lines, samples = np.asarray([[r["Line"], r["Sample"]] for r in res]).T
        else:
            lines, samples =  res["Line"], res["Sample"]
    except:
        raise Exception(res)

    return lines, samples


