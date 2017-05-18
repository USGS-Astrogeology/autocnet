import numpy as np
from scipy.spatial.distance import cdist

from autocnet.matcher.feature import FlannMatcher
from autocnet.matcher.feature_matcher import match
from autocnet.transformation.decompose import coupled_decomposition


def decompose_and_match(self, k=2, maxiteration=3, size=18, buf_dist=3,**kwargs):
    """
    Similar to match, this method first decomposed the image into
    $4^{maxiteration}$ subimages and applys matching between each sub-image.

    This method is potential slower than the standard match due to the
    overhead in matching, but can be significantly more accurate.  The
    increase in accuracy is a function of the total image size.  Suggested
    values for maxiteration are provided below.

    Parameters
    ----------
    k : int
        The number of neighbors to find

    method : {'coupled', 'whole'}
             whether to utilize coupled decomposition
             or match the whole image

    maxiteration : int
                   When using coupled decomposition, the number of recursive
                   divisions to apply.  The total number of resultant
                   sub-images will be 4 ** maxiteration.  Approximate values:

                    | Number of megapixels | maxiteration |
                    |----------------------|--------------|
                    | m < 10               |1-2|
                    | 10 < m < 30          | 3 |
                    | 30 < m < 100         | 4 |
                    | 100 < m < 1000       | 5 |
                    | m > 1000             | 6 |

    size : int
           When using coupled decomposition, the total number of points
           to check in each sub-image to try and find a match.
           Selection of this number is a balance between seeking a
           representative mid-point and computational cost.

    buf_dist : int
               When using coupled decomposition, the distance from the edge of
               the (sub)image a point must be in order to be used as a
               partioning point.  The smaller the distance, the more likely
               percision errors can results in erroneous partitions.
    """
    def func(group):
        ratio = 0.8
        res = [False] * len(group)
        if len(res) == 1:
            return [single]
        if group.iloc[0] < group.iloc[1] * ratio:
            res[0] = True
        return res

    # Grab the original image arrays
    sdata = self.source.get_array()
    ddata = self.destination.get_array()

    ssize = sdata.shape
    dsize = ddata.shape

    # Grab all the available candidate keypoints
    skp = self.source.get_keypoints()
    dkp = self.destination.get_keypoints()

    # Set up the membership arrays
    self.smembership = np.zeros(sdata.shape, dtype=np.int16)
    self.dmembership = np.zeros(ddata.shape, dtype=np.int16)
    self.smembership[:] = -1
    self.dmembership[:] = -1
    pcounter = 0

    # FLANN Matcher
    fl= FlannMatcher()

    for k in range(maxiteration):
        partitions = np.unique(self.smembership)
        for p in partitions:
            sy_part, sx_part = np.where(self.smembership == p)
            dy_part, dx_part = np.where(self.dmembership == p)

            # Get the source extent
            minsy = np.min(sy_part)
            maxsy = np.max(sy_part) + 1
            minsx = np.min(sx_part)
            maxsx = np.max(sx_part) + 1

            # Get the destination extent
            mindy = np.min(dy_part)
            maxdy = np.max(dy_part) + 1
            mindx = np.min(dx_part)
            maxdx = np.max(dx_part) + 1

            # Clip the sub image from the full images
            asub = sdata[minsy:maxsy, minsx:maxsx]
            bsub = ddata[mindy:maxdy, mindx:maxdx]

            # Utilize the FLANN matcher to find a match to approximate a center
            fl.add(self.destination.descriptors, self.destination['node_id'])
            fl.train()

            scounter = 0
            decompose = False
            while True:
                sub_skp = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx, minsy, maxsy))
                # Check the size to ensure a valid return
                if len(sub_skp) == 0:
                    break # No valid keypoints in this (sub)image
                if size > len(sub_skp):
                    size = len(sub_skp)
                candidate_idx = np.random.choice(sub_skp.index, size=size, replace=False)
                candidates = self.source.descriptors[candidate_idx]
                matches = fl.query(candidates, self.source['node_id'], k=3, index=candidate_idx)

                # Apply Lowe's ratio test to try to find a 'good' starting point
                mask = matches.groupby('source_idx')['distance'].transform(func).astype('bool')
                candidate_matches = matches[mask]
                match_idx = candidate_matches['source_idx'].astype(np.int)

                # Extract those matches that pass the ratio check
                sub_skp = skp.iloc[match_idx]

                # Check that valid points remain
                if len(sub_skp) == 0:
                    break

                # Locate the candidate closest to the middle of all of the matches
                smx, smy = sub_skp[['x', 'y']].mean()
                mid = np.array([[smx, smy]])
                dists = cdist(mid, sub_skp[['x', 'y']])
                closest = sub_skp.iloc[np.argmin(dists)]
                closest_idx = closest.name
                soriginx, soriginy = closest[['x', 'y']]

                # Grab the corresponding point in the destination
                q = candidate_matches.query('source_idx == {}'.format(closest.name))
                dest_idx = int(q['destination_idx'].iat[0])
                doriginx = dkp.at[dest_idx, 'x']
                doriginy = dkp.at[dest_idx, 'y']

                if mindy + buf_dist <= doriginy <= maxdy - buf_dist\
                 and mindx + 3 <= doriginx <= maxdx - 3:
                    # Point is good to split on
                    decompose = True
                    break
                else:
                    scounter += 1
                    if scounter >= maxiteration:
                        break

            # Clear the Flann matcher for reuse
            fl.clear()

            # Check that the identified match falls within the (sub)image
            # This catches most bad matches that have passed the ratio check
            if not (buf_dist <= doriginx - mindx <= bsub.shape[1] - buf_dist) or not\
                   (buf_dist <= doriginy - mindy <= bsub.shape[0] - buf_dist):
                   decompose = False

            if decompose:
                # Apply coupled decomposition, shifting the origin to the sub-image
                s_submembership, d_submembership = coupled_decomposition(asub, bsub,
                                                                     sorigin=(soriginx - minsx, soriginy - minsy),
                                                                     dorigin=(doriginx - mindx, doriginy - mindy),
                                                                     **kwargs)

                # Shift the returned membership counters to a set of unique numbers
                s_submembership += pcounter
                d_submembership += pcounter

                # And assign membership
                self.smembership[minsy:maxsy,
                            minsx:maxsx] = s_submembership
                self.dmembership[mindy:maxdy,
                            mindx:maxdx] = d_submembership
                pcounter += 4

    # Now match the decomposed segments to one another
    for p in np.unique(self.smembership):
        sy_part, sx_part = np.where(self.smembership == p)
        dy_part, dx_part = np.where(self.dmembership == p)

        # Get the source extent
        minsy = np.min(sy_part)
        maxsy = np.max(sy_part) + 1
        minsx = np.min(sx_part)
        maxsx = np.max(sx_part) + 1

        # Get the destination extent
        mindy = np.min(dy_part)
        maxdy = np.max(dy_part) + 1
        mindx = np.min(dx_part)
        maxdx = np.max(dx_part) + 1

        # Get the indices of the candidate keypoints within those regions / variables are pulled before decomp.
        sidx = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx, minsy, maxsy)).index
        didx = dkp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(mindx, maxdx, mindy, maxdy)).index
        # If the candidates < k, OpenCV throws an error
        if len(sidx) >= k and len(didx) >=k:
            match(self, aidx=sidx, bidx=didx)
            match(self, aidx=didx, bidx=sidx)
