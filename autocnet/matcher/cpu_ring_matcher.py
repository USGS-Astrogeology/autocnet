import numpy as np

def check_pidx_duplicates(pidx):
    """
    Given a ring match generted set of indices, 
    apply outlier detection to ensure no duplicates
    exist in either the reference column or the
    source column. If duplicates do exist, remove the 
    rows; the solution is ambiguous.
    """
    # Check for duplicates
    l = pidx[:,1].tolist()
    clean = [i for i, x in enumerate(l) if l.count(x) == 1]
    pidx = pidx[clean, :]

    l = pidx[:,0].tolist()
    clean = [i for i, x in enumerate(l) if l.count(x) == 1]
    pidx = pidx[clean, :]
    return pidx

def ransac_permute(ref_points, tar_points, tolerance_val, target_points):
    """
    Given a set of reference points and target points, compute the
    geometric distances between pairs of points in the reference set
    and pairs of points in the target set.  Points for which the ratio of
    the distances is within plus or minus 1 - tolerance_value are considered
    to be good matches.

    If a valid solution is not found, this func returns three empty lists.

    Parameters
    ----------
    ref_points : ndarray
                 A (n, 2) array of points where the first column is the x
                 pixel location and the second column is the y pixel location.
                 Additional columns are ignored. These points are from one
                 image.

    tar_points : ndarray
                 A (n,2) array as above, for a second image.

    tolerance_value : float
                      On the range [-1, 1], the computed ratio must be
                      within 1 +- tolerance to be considered a match

    target_points : int
                    The minimum number of points required to return
                    a valid answer

    Returns
    -------
    ref_points : ndarray
                 (n,2) subset of the input ref_points

    tar_points : ndarray
                 (n,2) subset of the input tar points

    f2 : ndarray
         of indices for valid points

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """
    n = len(ref_points)
    # Build an n,n distance matrix to store pairwise point distances
    dist = np.zeros((n,n))
    for i in range(n):
        vr1 = ref_points[i]
        vt1 = tar_points[i]
        for j in range(i, n):
            # These are diagonals so always zero
            if i == j:
                dist[i,j] = 0
                dist[j,i] = 0
                continue
            # Compute the distance between ref_b - ref_a and tar_b - tar_a. The
            # absolute value should be small as these points should be sp
            vr2 = ref_points[j]
            vt2 = tar_points[j]

            dr = vr2 - vr1
            dt = vt2 - vt1

            dist[i,j] = (dr[0]**2 + dr[1]**2)**0.5 / (dt[0]**2+dt[1]**2)**0.5
            dist[j,i] = dist[i,j]
    # Compute min/max bounds for the tolerance
    minlim = 1 - tolerance_val
    maxlim = 1 + tolerance_val

    # Determine which points are within the tolerance
    q1 = dist >= minlim
    q2 = dist <= maxlim
    q = (q1*q2).astype(np.int)
    # How many points are within the tolerance?
    s = np.sum(q, axis=1)
    # If the number of points within the tolerance are greater than the number of desired points
    if np.max(s) >= target_points:
        m = np.eye(n).dot(target_points + 1)
        for i in range(n):
            for j in range(i):
                m[i,j] = q[i].dot(q[j])
                m[j,i] = m[i,j]
        qm = m >= target_points
        sqm = np.sum(qm, axis=-1)
        f = np.argmax(sqm)
        f2 = np.nonzero(qm[f])
        return ref_points[f2], tar_points[f2], f2
    else:
        return [], [], []

def sift_match(a, b, thresh=1.5):
    """
    vl_ubcmatch from the vlfeat toolbox for MatLab.  This is
    Lowe's prescribed implementation for disambiguating descriptors.

    Parameters
    ----------
    a : np.ndarray
        (m,) a singular descriptors where the m-dimension are the
        descriptor lengths.  For SIFT m=128. This is reshaped from
        a vector to an array.
    b : np.ndarray
        (n,m) where the n-dimension are the individual features and
        the m-dimension are the elements of the descriptor.

    thresh : float
             The threshold for disambiguating correspondences. From Lowe.
             If best * thresh < second_best, a match has been found.

    Returns
    -------
    best : int
           Index for the best match

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """
    a = a.reshape(1,-1)
    dists = np.sum((b-a)**2, axis=1)

    try:
        best = np.nanargmin(dists)
    except:
        return

    if len(dists[dists != dists[best]]) == 0:
        return  # Edge case where all descriptors are the same
    elif len(dists[dists == dists[best]]) > 1:
        return  # Edge case where the best is ambiguous
    sec_best = np.nanmin(dists[dists != dists[best]])

    if dists[best] * thresh < sec_best:
        return best
    return

def ring_match(ref_feats, tar_feats, ref_desc, tar_desc, ring_radius=4000,
               max_radius=40000, target_points=15, tolerance_val=0.02,
               iteration_break_point=200):
    """
    Apply the University College London ring matching technique that seeks to match
    target feats to a number of reference features.

    Parameters
    ----------
    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    ring_radius : numeric
                  The width of a ring for matching. In the same units as the x,y
                  coordinates for the features, e.g. if the ref_feats and tar_feats
                  are provided in pixel space and meters, the ring_radius should
                  be expressed in meters

    max_radius : numeric
                 The maximum radius to be tested.  This is the maxixum distance
                 a given correspondence could be from the initial estimate.

    target_points : int
                    The number of points that constitute a valid match

    tolerance : float
                The tolerance for outlier detection in point ordering between estimated
                resolutions

    Returns
    -------
    xref : ndarray
           (n,4) array of the correspondences selected from the ref_feats input

    xtar : ndarray
           (n,4) array of the correspondences selected from the tar_feats input

    p_idx : ndarray
            (n,2) array of the indices in the ref_feats and tar_feats input arrays

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """

    # Reference and target features
    ref_xy = ref_feats[:,:2]
    ref_xmym = ref_feats[:,2:4]
    tar_xy = tar_feats[:,:2]
    tar_xmym = tar_feats[:,2:4]
    
    # Boolean mask for those reference points that have already been matched
    ref_mask = np.ones(len(ref_xy), dtype=bool)
 
    # Counters
    numr = len(ref_feats)
    rad_num = int(max_radius / ring_radius) # Number of radial rings
    points_num = np.zeros(rad_num, dtype=np.int)  # Number of points per ring vector
    metr = 1
    
    # Initial array for holding candidate points
    p = np.zeros((target_points, 4 * rad_num))
    p_idx = np.zeros((target_points, 2 * rad_num), dtype=np.int)

    while ref_mask.any():
        # Grab a random reference point
        r = np.random.choice(np.arange(numr)[ref_mask])
        current_ref_desc = ref_desc[r]
        current_ref_xy = ref_xy[r]
        current_ref_xmym = ref_xmym[r]
        # Compute the euclidean distance between the reference point and all targets
        d = np.linalg.norm(current_ref_xmym - tar_xmym, axis=1)
        # For each point, independently match to a point in a given ring
        for i in range(rad_num):
            # The number of points that are within a given ring
            z = points_in_ring(d, i*ring_radius, (i+1) * ring_radius)
            # If we have enough points, run the sift matcher and select the best point, updating p
            if np.sum(z) >= target_points:
                # All candidate points that are in the ring
                current_tar_descs = tar_desc[z]  # This slicing uses ~25% of processing timr
                current_tar_xys = tar_xy[z]
                z_idx = np.where(z == True)[0]
                # Sift Match
                match = sift_match(current_ref_desc, current_tar_descs, thresh=1.5)  # The remaining 75% of processing time.
                if match is not None:
                    if points_num[i] == p.shape[0]:
                        p = dynamically_grow_array(p, target_points)
                        p_idx = dynamically_grow_array(p_idx, target_points)
                    p[points_num[i], 4*i:4*i+4] = [current_ref_xy[0], current_ref_xy[1], current_tar_xys[match][0], current_tar_xys[match][1]]
                    # Set the id of the point
                    p_idx[points_num[i], 2*i:2*i+2] = [r, z_idx[match]]
                    points_num[i] += 1
        
        if (metr % iteration_break_point == 0) or (np.sum(ref_mask) == 0):
            max_cons = 3
            # Find all candidate rings
            candidate_rings = points_num >= target_points
            for j in range(rad_num):
                if candidate_rings[j]:
                    # For each ring that is a candidate, select all of the reference and targets from the p matrix
                    # the first part of the slice grabs all candidate points and the second half in the first two args
                    # selects the ref and target (respectively).
                    npoints_in_ring = points_num[j]
                    ref_points = p[:npoints_in_ring, 4*j:4*j+2] # Slice out the reference coords
                    tar_points = p[:npoints_in_ring, 4*j+2:4*j+4]  # Slice out the target coords
                    indices = p_idx[:npoints_in_ring, 2*j:2*j+2] # slice out the indices

                    xref, xtar, idx = ransac_permute(ref_points, tar_points, tolerance_val, target_points)
                    # This selects the best of the rings
                    max_cons = max(max_cons, len(xref))
                    if len(xref) >= target_points:
                        # Solution found
                        # Instead of returning the ref and tar points, return the ids
                        ring = (j*ring_radius, j*ring_radius+ring_radius)
                        return xref, xtar, indices[idx], ring
        # Mask the reference point and iterate to the next randomly selected point
        ref_mask[r] = False
        metr += 1
    return None, None, None, 'Exhausted'

def points_in_ring(distance_vector, inner_radius, outer_radius):
    """
    Returns the indices of all points within a given distance from
    a distance vector. The vector is assumed to be 1d.

    Parameters
    ----------
    distance_vector : ndarray
                      (n,1) array of distances between a point and a set of points
    inner_radius : float
                   The lower threshold for the ring
    outer_radius : float
                   The outer threshold for the ring

    Returns
    -------
     : ndarray
       (n,1) array of booleans where all correspondences inside of the array are True
    """
    return (distance_vector >= inner_radius) * (distance_vector <= outer_radius)

def dynamically_grow_array(array, m):
    """
    Given an array, dynamically grow the array vertically with an m,n array of zeros

    Parameters
    ----------
    array : ndarray
            The array to be grown
    m : int
        The number of new rows to add

    Returns
    -------
    array : ndarray
            The expanded array
    """
    array_append = np.zeros((m, array.shape[1]))
    array = np.vstack((array, array_append))

    return array


def directed_ring_match(ref_feats, tar_feats, ref_desc, tar_desc, ring_min, ring_max, target_points=15, tolerance_value=0.02):
    """
    Given an input set of reference features and target features, attempt to find
    correspondences within a given ring, where the ring is defined by min and max
    radii.  This is a directed version of the ring_match function, in that this
    function assumes that the correspondence is within the defined ring.

    This implementation is inspired by and uses the ring_match implementation above, developed
    by Sidiropoulos and Muller.

    Parameters
    ----------
    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    ring_min : numeric
               The inner distance of the ring

    ring_max : numeric
               The outer distance of the ring


    target_points : int
                    The number of points that constitute a valid match

    tolerance : float
                The tolerance for outlier detection in point ordering between estimated
                resolutions

    Returns
    -------
    xref : ndarray
           (n,4) array of the correspondences selected from the ref_feats input

    xtar : ndarray
           (n,4) array of the correspondences selected from the tar_feats input

    p_idx : ndarray
            (n,2) array of the indices in the ref_feats and tar_feats input arrays
    """

    # Reference and target features
    ref_xy = ref_feats[:,:2]
    ref_xmym = ref_feats[:,3:]
    tar_xy = tar_feats[:,:2]
    tar_xmym = tar_feats[:,3:]

    numr = len(ref_feats)

    p = np.zeros((numr, 4))
    p_idx = np.zeros((numr, 2), dtype=np.int)
    points_num = 0
    # Iterate over all of the reference points seeking a match
    for r in range(numr):
        current_ref_desc = ref_desc[r]
        current_ref_xy = ref_xy[r]
        current_ref_xmym = ref_xmym[r]

        # Compute the euclidean distance between the reference point and all targets
        d = np.linalg.norm(current_ref_xmym - tar_xmym, axis=1)
        # Find all of the candidates, if none, skip
        z = (d > ring_min) * (d < ring_max)

        if np.sum(z) == 0:
            continue
        current_tar_descs = tar_desc[z]
        current_tar_xys = tar_xy[z]
        z_idx = np.where(z == True)[0]

        match = sift_match(current_ref_desc, current_tar_descs, thresh=1.5)
        if match is not None:
            p[points_num] = [current_ref_xy[0], current_ref_xy[1], current_tar_xys[match][0], current_tar_xys[match][1]]
            # Set the id of the point
            p_idx[points_num] = [r, z_idx[match]]
            points_num += 1
    if points_num == 0:
        # No candidate matches found in this set.
        return [], [], []
    # Now that the candidates have all been located, check their geometric relationships to find the good matches
    ref_points = p[:points_num, :2]
    tar_points = p[:points_num, 2:]
    xref, xtar, idx = ransac_permute(ref_points, tar_points, tolerance_value, target_points)
    return xref, xtar, p_idx[idx]

def ring_match_one(x, y, ref_feats, tar_feats, ref_desc, tar_desc, ring, search_radius=600, max_search_radius=600, target_points=5):
    """
    Given an x,y coordinate where a match is desired, find all candidates within
    some search radius and attempt to generate a match.  If no matches are identified
    expand the search radius by search_radius and search again. Once a match
    is identified (by calling the directed matcher and finding a geometric consensus
    with at least target_points), the match closest to the given x,y is returned.

    Parameters
    ----------
    x : int
        x coordinate in image space

    y : int
        y coordinate in image space

    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    ring : tuple
           in the form (min_ring, max_ring)

    search_radius : numeric
                    The radius of the search extent in image pixels

    target_points : int
                    The desired number of points to identify a correspondence
    """
    search = True

    # Start search within search_radius of the center of the cell
    while search_radius <= max_search_radius:
        # Find all correspondences within 100m of the center of the cell to fill
        candidates = np.where((np.abs(ref_feats[:,0]-y) < search_radius) &\
                              (np.abs(ref_feats[:,1]-x) < search_radius))[0]

        sref_feats = ref_feats[candidates]
        sref_desc = ref_desc[candidates]

        min_ring = ring[0]
        max_ring = ring[1]

        sx_ref, sx_tar, sindices, = directed_ring_match(sref_feats, tar_feats, sref_desc, tar_desc, min_ring, max_ring, target_points=target_points)
        if len(sindices) >= target_points:

            #sindices[:,0] = candidates[sindices[:,0]]

            # Find the match that is closest to the center of the cell.
            center = np.array([x, y])
            ref_coords = sref_feats[sindices[:,0], :2]
            dist = np.linalg.norm(center - ref_coords, axis=1)
            closest_idx = np.argmin(dist)

            # Remap the reference from the subset to the full set
            sindices[:,0] = candidates[sindices[:,0]]

            return sindices[closest_idx]

        # expand the search radius by 100m
        search_radius += search_radius
        if search_radius >= max_search_radius:
            return []

def add_correspondences(in_feats, ref_feats, tar_feats, ref_desc, tar_desc,  xextent, yextent, ring, n_x_cells=5, n_y_cells=5, target_points=5, **kwargs):
    """
    Given a set of input features and x/y extents lay a regular grid over the area
    defined by the extents and find correspondences within each grid cell that
    does not have any existing correspondences.

    Then number of cells are defined by the n_x/y_cells parameters.

    Parameters
    ----------
    in_feats : ndarray
               (n,m) input features with the first column being x-coordinate
               in image space and the second column being y-coordinate.

    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    xextent : tuple
              in the form (minx, maxx)

    yextent : tuple
              in the form (miny, maxy)

    ring : tuple
           in the form (min_ring, max_ring)

    n_x_cells : int
                the number of cells to generate in the x direction

    n_y_cells : int
                the number of cells to generate in the y direction

    target_points : int
                    The desired number of points to identify a correspondence
    """
    x_edges = np.linspace(xextent[0], xextent[1], n_x_cells)
    y_edges = np.linspace(yextent[0], yextent[1], n_y_cells)

    # Find the cells that are populated and assign as covered
    xbins = np.digitize(in_feats[:,1], bins=x_edges)
    ybins = np.digitize(in_feats[:,0], bins=y_edges)
    covered = list(zip(xbins, ybins))

    refs_to_add = []
    # Loop over all cells
    for i in range(1, n_x_cells):
        for j in range(1, n_y_cells):
            # and process only the uncovered cells
            if (i,j) in covered:
                continue
            x_cell_center = x_edges[i-1] + (x_edges[i] - x_edges[i-1])/2
            y_cell_center = y_edges[j-1] + (y_edges[j] - y_edges[j-1])/2

            ref = ring_match_one(x_cell_center, y_cell_center, ref_feats, tar_feats, ref_desc, tar_desc, ring, **kwargs)
            refs_to_add.append(ref)
    return refs_to_add
