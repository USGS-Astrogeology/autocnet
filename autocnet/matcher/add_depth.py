import numpy as np

from autocnet.utils.utils import normalize_vector


def deepen_correspondences(ab_kp, bc, source_idx,
                           clean_keys=['fundamental'],
                           geometric_threshold=2):
    """
    Given a set of input correspondences, use the fundamental matrix to search
    for additional correspondences.

    The algorithm functions by selecting all edges incident to the given node,
    concatenating the dataframes of matches into a single large table, and then
    grouping those matches by the current node's correspondence index.  In an
    idealized case, the number of entries in each group would equal the number
    of incident edges.  When this is not true, the point in the source image
    is projected to the epipolar line in the destination image and a search
    of previously omitted points is performed.  Should a previously omitted
    point fulfill the geometric constraint, the match is added to the
    currently valid set.

    Parameters
    ----------
    ab_kp : ndarray
            Homogeneous point that is projected to an epipolar line in bc

    bc : object
         Edge object with points that are searched along
         the epipolar line defined by ab

    source_idx : int
                 Index into bc identifying candidate matches

    geometric_threshold : float
                          The maximum projection error, in pixels, a point can be
                          from the corresponding epipolar line to still be considered
                          an inlier.
    """

    # Grab the edge and the edge candidate coordinates
    bc_x = np.empty((bc.destination.nkeypoints, 3))
    bc_x[:, -1] = 1.0
    bc_x[:, :2] = bc.destination.get_keypoint_coordinates().values

    # Grab F for reprojection
    f_matrix = bc.fundamental_matrix

    # Compute the epipolar line projecting point ab into bc
    epipolar_line = normalize_vector(ab_kp.dot(f_matrix.T))

    # Check to see if a previously removed candidate fulfills the threshold geometric constraint
    bc_candidates = bc.matches[(bc.matches['source_idx'] == source_idx)]
    bc_candidate_coords = np.empty((len(bc_candidates), 3))
    bc_candidate_coords[:, -1] = 1.
    bc_candidate_coords[:, :2] = bc.destination.get_keypoint_coordinates(index=bc_candidates['destination_idx']).values
    bc_distance = np.abs(epipolar_line.dot(bc_candidate_coords.T))

    # Get the matches
    second_order_candidates = np.where(bc_distance < geometric_threshold)[0]

    # In testing, every single valid second order candidate has a single, duplicated entry.
    # That is, the correspondence has passed symmetry, but failed some other check.  Therefore,
    # an additional descriptor distance check is omitted here.
    if len(second_order_candidates) > 0:
        # Update the mask to include this new point
        new_match = bc_candidates.iloc[second_order_candidates[0]]
        coords = bc_candidate_coords[second_order_candidates[0]]
        return coords, new_match.name
    else:
        return None, None
