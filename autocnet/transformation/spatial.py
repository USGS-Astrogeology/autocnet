import pyproj

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
    source_pyproj = pyproj.Proj(proj = source_proj, a = semi_major, b = semi_minor)
    dest_pyproj = pyproj.Proj(proj = dest_proj, a = semi_major, b = semi_minor)

    y, x, z = pyproj.transform(source_pyproj, dest_pyproj, record[0], record[1], record[2], **kwargs)
    return y, x, z


def match_mosaic(image, mosaic, cnet):
    """
    Matches an image node with a image mosaic given a control network.

    Parameters
    ----------
    image : str
            Path to projected cube. The projection must match the mosaic image's
            projection and should intersect with input mosaic

    mosaic : Geodataset
             Mosaic geodataset

    cnet : DataFrame
           Control network dataframe, output of plio's from_isis

    Returns
    -------
    : DataFrame
      DataFrame containing source points containing matched features.

    : list
      List in the format [minline,maxline, minsample,maxsample], the line/sample
      extents in the mosaic matching the image

    """
    cube = GeoDataset(image)

    # Get lat lons from body fixed coordinates
    a = elysium.metadata["IsisCube"]["Mapping"]["EquatorialRadius"]
    b = elysium.metadata["IsisCube"]["Mapping"]["PolarRadius"]
    ecef = pyproj.Proj(proj='geocent', a=a, b=b)
    lla = pyproj.Proj(proj='latlon', a=a, b=b)
    lons, lats, alts = pyproj.transform(ecef, lla, np.asarray(cnet["adjustedX"]), np.asarray(cnet["adjustedY"]), np.asarray(cnet["adjustedZ"]))
    gdf = geopandas.GeoDataFrame(cnet, geometry=geopandas.points_from_xy(lons, lats))
    points = gdf.geometry


    # get footprint
    image_arr = cube.read_array()

    footprint = wkt.loads(cube.footprint.ExportToWkt())

    # find ground points from themis cnet
    spatial_index = gdf.sindex
    possible_matches_index = list(spatial_index.intersection(footprint.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(footprint)]
    points = precise_matches.geometry

    pts = unique_rows(np.asarray([cube.latlon_to_pixel(point.y, point.x) for point in points]))

    image_pts = []
    for p in pts:
        if not np.isclose(image_arr[p[1], p[0]], ISISNULL):
            image_pts.append(p)
    image_pts = np.asarray(image_pts)

    minlon, minlat, maxlon, maxlat = footprint.bounds

    samples, lines = np.asarray([mosaic.latlon_to_pixel(p[0],p[1])for p in [[minlat, minlon], [maxlat, maxlon]]]).T
    minline = min(lines)
    minsample = min(samples)
    maxline = max(lines)
    maxsample = max(samples)

    # hard code for now, we should get type from label
    mosaic_arr = mosaic.read_array().astype(np.uint8)
    sub_mosaic = mosaic_arr[minline:maxline, minsample:maxsample]

    image_arr[np.isclose(image_arr, ISISNULL)] = np.nan
    match_results = []
    for k, p in enumerate(image_pts):
        sx, sy = p

        try:
            ret = refine_subpixel(sx, sy, sx, sy, image_arr, sub_mosaic, size=10, reduction=1, convergence_threshold=1)
        except Exception as ex:
            continue

        if ret is not None:
            x,y,metrics = ret
        else:
            continue

        dist = np.linalg.norm([x-dx, y-dy])
        match_results.append([points.index[k], x-dx, y-dy, dist, p[0] ,p[1]])

    match_results = pd.DataFrame(match_results, columns=["cnet_index", "x_offset", "y_offset", "dist", "x", "y"])

    return match_results, [minline,maxline, minsample,maxsample]
