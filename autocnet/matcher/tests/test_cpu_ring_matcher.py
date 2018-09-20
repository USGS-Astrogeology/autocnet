from unittest import mock

import numpy as np
import pytest

from autocnet.matcher import cpu_ring_matcher as rm

@pytest.mark.parametrize('arr, expected', [
                         (np.array([[1,0],[1,1], [2,3]]), (1,2)),
                         (np.array([[0,0], [1,1], [2,2]]), (3,2)
                        )])
def test_check_pidx_duplicates(arr, expected):
    pidx = rm.check_pidx_duplicates(arr)
    assert pidx.shape == expected

@pytest.mark.parametrize("a, b, threshold, expected", [
                         # Tests standard call
                         (np.array([1,2,3]), 
                          np.array([[1,2,3], [4,5,6], [7,8,9]]),
                          1.5, 
                          np.array([0])),
                          # Tests call where distances are too close
                        (np.array([1,2,3]),
                         np.array([[7,8,9], [1,2,4], [1,2,4.1]]),
                         1.5, 
                         None),
                         # Tests call with close distances where the threshold is low
                        (np.array([1,2,3]),
                         np.array([[7,8,9], [1,2,4], [1,2,4.1]]),
                         1., 
                         1),
                         # Tests call when np.argmin will fail
                        (np.array([np.nan, np.nan]),
                         np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                         1.5,
                         None),
                         # Tests call where descriptors are identical
                        (np.array([1,2,3]),
                         np.array([[1,2,3], [1,2,3], [1,2,3]]),
                         1.5,
                         None)
])
def test_sift_match(a, b, threshold, expected):
    assert rm.sift_match(a, b, thresh=threshold) == expected    

@pytest.mark.parametrize("x,y, eidx",[(np.array([[1,1],[2,2],[3,3], [4,4], [5,5]]),
                                       np.array([[1.1,1.0],[1.9,1.95],[3,3], [-4,-4], [5,5]]),
                                       np.array([[0,1,2,4]])),
                                      (np.array([[1,1], [5,5]]),
                                       np.array([[1,1], [3,3]]),
                                       [])
                                      ])
def test_ransac_permute(x, y, eidx):
    xp, yp, idx = rm.ransac_permute(x, y, 0.2, 2)
    np.testing.assert_array_equal(idx, eidx)


def test_add_correspondences():
    func = 'autocnet.matcher.cpu_ring_matcher.ring_match_one'
    with mock.patch(func, return_value=1):
        in_feats = np.array([[1,1], [2,2]])
        ref_feats = np.array([[1,1],[2,2],[3,3], [4,4], [5,5]])
        tar_feats = np.array([[1.1,1.0],[1.9,1.95],[3,3], [-4,-4], [5,5]])
        
        rm.add_correspondences(in_feats, ref_feats, tar_feats, None, None,
                               (0,6), (0,6),(0,1))

def test_dynamically_grow():
    x = np.ones((3,3))
    y = rm.dynamically_grow_array(x,6)
    assert y.shape == (9,3)
    
def test_dynamically_grow_dtype():
    x = np.ones((3,3), dtype=np.int8)
    y = rm.dynamically_grow_array(x,6)
    assert np.issubdtype(y.dtype, np.float64)

    y = rm.dynamically_grow_array(x,6,dtype=np.int8)
    assert np.issubdtype(y.dtype, np.int8)

def test_points_in_ring():
    x = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
    for i in np.arange(0.5, 4.5):
        assert np.sum(rm.points_in_ring(x, i, i+1)) == 5

def test_ring_match():
    ref_feats = np.array([[1,1,1,1],
                          [2,2,2,2],
                          [3,3,3,3],
                          [4,4,4,4]])
    tar_feats = np.array([[2,2,1.1,1],
                          [2.5, 2.5, 1.1, 1.1],
                          [3,3,2.1,2.1],
                          [3.5, 3.5, 2.2, 2.2],
                          [4,4,2.9,2.9],
                          [4.5, 4.5, 3.0, 3.0],
                          [5,5, 4.0, 4.1],
                          [5.5, 5.5, 4.1, 4.1]])
    ref_desc = np.array([[0,0,0,0],
                         [1,1,1,1],
                         [2,2,2,2],
                         [3,3,3,3]])
    tar_desc = np.array([[0,0,0,0],
                         [6,7,8,9],
                         [1,1,1,1],
                         [6,7,8,9],
                         [2,2,2,2],
                         [6,7,8,9],
                         [3,3,3,3],
                         [6,7,8,9]])

    ring_radius = 0.5
    max_radius = 1
    target_points = 2
    tolerance = 0.1
    gr, gt, p_idx, ring = rm.ring_match(ref_feats, tar_feats, ref_desc, tar_desc,
                                     ring_radius=ring_radius, max_radius=max_radius,
                                     target_points=target_points, tolerance_val=tolerance,
                                     iteration_break_point=2)
    assert ring == (0.0, 0.5)
    sorted_pidx = p_idx[p_idx[:,0].astype(np.int).argsort()]
    np.testing.assert_array_equal(sorted_pidx,
                                  np.array([[0,0],[1,2],[2,4],[3,6]]))
