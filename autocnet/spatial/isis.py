"""This module contains wrappers around the ISIS campt and mappt that are
specific to understanding the relationship between pixels and projected
coordinates."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import os
from collections import abc
from numbers import Number

import numpy as np
import kalasiris as isis
import pvl


def point_info(
        cube_path: os.PathLike,
        x,
        y,
        point_type: str,
        allowoutside=False
):
    """
    Returns a pvl.collections.MutableMappingSequence object or a
    Sequence of MutableMappingSequence objects which contain keys
    and values derived from the output of ISIS campt or mappt on
    the *cube_path*.

    If x and y are single numbers, then a single MutableMappingSequence
    object will be returned.  If they are Sequences or Numpy arrays, then a
    Sequence of MutableMappingSequence objects will be returned,
    such that the first MutableMappingSequence object of the returned
    Sequence will correspond to the result of *x[0]* and *y[0]*,
    etc.

    Raises subprocess.CalledProcessError if campt or mappt have failures.
    May raise ValueError if campt completes, but reports errors.

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    x : Number, Sequence of Numbers, or Numpy Array
        Point(s) in the x direction. Interpreted as either a sample
        or a longitude value determined by *point_type*.

    y : Number, Sequence of Numbers, or Numpy Array
        Point(s) in the y direction. Interpreted as either a line
        or a latitude value determined by *point_type*.

    point_type : str
                 Options: {"image", "ground"}
                 Pass "image" if  x,y are in image space (sample, line) or
                 "ground" if in ground space (longitude, latitude)

    allowoutside: bool
                  Defaults to False, this parameter is passed to campt
                  or mappt.  Please read the ISIS documentation to
                  learn more about this parameter.

    """
    point_type = point_type.casefold()
    valid_types = {"image", "ground"}
    if point_type not in valid_types:
        raise ValueError(
            f'{point_type} is not a valid point type, valid types are '
            f'{valid_types}'
        )

    if isinstance(x, abc.Sequence) and isinstance(y, abc.Sequence):
        if len(x) != len(y):
            raise IndexError(
                f"Sequences given to x and y must be of the same length."
            )
        x_coords = x
        y_coords = y
    elif isinstance(x, Number) and isinstance(y, Number):
        x_coords = [x, ]
        y_coords = [y, ]
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if not all((x.ndim == 1, y.ndim == 1)):
            raise IndexError(
                f"If they are numpy arrays, x and y must be one-dimensional, "
                f"they were: {x.ndim} and {y.ndim}"
            )
        if x.shape != y.shape:
            raise IndexError(
                f"Numpy arrays given to x and y must be of the same shape."
            )
        x_coords = x
        y_coords = y
    else:
        raise TypeError(
            f"The values of x and y were neither Sequences nor individual "
            f"numbers, they were: {x} and {y}"
        )

    if point_type == "image":
        # convert to ISIS pixels
        for i in range(len(x_coords)):
            x_coords[i] += 0.5
            y_coords[i] += 0.5

    results = []
    if pvl.load(cube_path).get("IsisCube").get("Mapping"):
        # We have a projected image, and must use mappt
        mappt_common_args = dict(allowoutside=allowoutside, type=point_type)

        for xx, yy in zip(x_coords, y_coords):
            mappt_args = {
                "ground": dict(
                    longitude=xx,
                    latitude=yy,
                    coordsys="UNIVERSAL"
                ),
                "image": dict(
                    sample=xx,
                    line=yy,
                )
            }
            for k in mappt_args.keys():
                mappt_args[k].update(mappt_common_args)

            results.append(pvl.loads(
                isis.mappt(cube_path, **mappt_args[point_type]).stdout
            )["Results"])
    else:
        # Not projected, use campt
        if point_type == "ground":
            # campt uses lat, lon for ground but sample, line for image.
            # So swap x,y for ground-to-image calls
            p_list = [f"{lat}, {lon}" for lon, lat in zip(x_coords, y_coords)]
        else:
            p_list = [
                f"{samp}, {line}" for samp, line in zip(x_coords, y_coords)
            ]

        # ISIS's campt needs points in a file
        with isis.fromlist.temp(p_list) as f:
            cp = isis.campt(
                cube_path,
                coordlist=f,
                allowoutside=allowoutside,
                usecoordlist=True,
                coordtype=point_type
            )

        camres = pvl.loads(cp.stdout)
        for r in camres.getall("GroundPoint"):
            if r['Error'] is None:
                # convert all pixels to PLIO pixels from ISIS
                r["Sample"] -= .5
                r["Line"] -= .5
                results.append(r)
            else:
                raise ValueError(
                    f"ISIS campt completed, but reported an error: {r['Error']}"
                )

    if isinstance(x, (abc.Sequence, np.ndarray)):
        return results
    else:
        return results[0]


def image_to_ground(
        cube_path: os.PathLike,
        sample,
        line,
        lontype="PositiveEast360Longitude",
        lattype="PlanetocentricLatitude",
):
    """
    Returns a two-tuple of numpy arrays or a two-tuple of floats, where
    the first element of the tuple is the longitude(s) and the second
    element are the latitude(s) that represent the coordinate(s) of the
    input *sample* and *line* in *cube_path*.

    If *sample* and *line* are single numbers, then the returned two-tuple
    will have single elements. If they are Sequences, then the returned
    two-tuple will contain numpy arrays.

    Raises the same exceptions as point_info().

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    sample : Number or Sequence of Numbers
        Sample coordinate(s).

    line : Number or Sequence of Numbers
        Line coordinate(s).

    lontype: str
        Name of key to query in the campt or mappt return to get the returned
        longitudes. Defaults to "PositiveEast360Longitude", but other values
        are possible. Please see the campt or mappt documentation.

    lattype: str
        Name of key to query in the campt or mappt return to get the returned
        latitudes. Defaults to "PlanetocentricLatitude", but other values
        are possible.  Please see the campt or mappt documentation.

    """
    res = point_info(cube_path, sample, line, "image")

    if isinstance(sample, (abc.Sequence, np.ndarray)):
        lons, lats = np.asarray([
            [r[lontype], r[lattype]] for r in res
        ]).T
    else:
        lons, lats = res[lontype].value, res[lattype].value

    return lons, lats


def ground_to_image(cube_path, lon, lat):
    """
    Returns a two-tuple of numpy arrays or a two-tuple of floats, where
    the first element of the tuple is the sample(s) and the second
    element are the lines(s) that represent the coordinate(s) of the
    input *lon* and *lat* in *cube_path*.

    If *lon* and *lat* are single numbers, then the returned two-tuple
    will have single elements. If they are Sequences, then the returned
    two-tuple will contain numpy arrays.

    Raises the same exceptions as point_info().

    Parameters
    ----------
    cube_path : os.PathLike
                Path to the input cube.

    lon: Number or Sequence of Numbers
        Longitude coordinate(s).

    lat: Number or Sequence of Numbers
        Latitude coordinate(s).

    """
    res = point_info(cube_path, lon, lat, "ground")

    if isinstance(lon, (abc.Sequence, np.ndarray)):
        samples, lines = np.asarray([[r["Sample"], r["Line"]] for r in res]).T
    else:
        samples, lines = res["Sample"], res["Line"]

    return samples, lines


