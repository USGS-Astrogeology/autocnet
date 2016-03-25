import numpy as np
import networkx as nx


def mcl(g, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
    """
    Markov Cluster Algorithm

    Implementation modified from: https://github.com/koteth/python_mcl
    Originally released under the MIT license (https://opensource.org/licenses/MIT)

    Parameters
    ----------
    g : object or ndarray
        NetworkX graph object or adjacency matrix

    inflate_factor : float
                     Parameter to strengthen and weaken flow between nodes.  The larger the value
                     the more granular the resultant clusters are.

    expand_factor : int
                    Parameter to manage flow connection between different regions of the graph.

    mult_factor : int
                  Value to set for self loops.  That is, the flow between a node and itself.

    max_loop : int
               Number of iterations to perform before terminating (or convergence).

    Returns
    -------
    arr : ndarray
          arr normalized flow matrix computed after convergence or max_loop is exceeded.

    clusters : dict
               of clusters where the key is an arbitrary cluster identifier and
               the value is a list of node identifiers.

    References
    ----------
    [Stijn2000]_
    [Stijn2000a]_

    """

    def _normalize(arr):
        """
        Column normalize an array
        Parameters
        ----------
        arr : ndarray
              array to be normalized
        Returns
        -------
        new_matrix : ndarray
                     normalized array
        """
        column_sums = arr.sum(axis=0)
        new_matrix = arr / column_sums[np.newaxis, :]
        return new_matrix

    def _inflate(arr, inflate_factor):
        return _normalize(np.power(arr, inflate_factor))

    def _expand(arr, expand_factor):
        return np.linalg.matrix_power(arr, expand_factor)

    def _add_diag(arr, mult_factor):
        return arr + mult_factor * np.identity(arr.shape[0])

    def _stop(arr, i):

        if i % 5 == 4:
            m = np.max(arr ** 2 - arr) - np.min(arr ** 2 - arr)
            if m == 0:
                return True

        return False

    def _get_clusters(arr):
        clusters = {}
        cid = 0
        for j in arr:
            row_positive = np.nonzero(j)[0].tolist()
            if row_positive and not row_positive in clusters.values():
                clusters[cid] = row_positive
                cid += 1
        return clusters

    # Create a dense adjacency matrix
    if isinstance(g, nx.Graph):
        arr = np.array(nx.adjacency_matrix(g).todense())
    elif isinstance(g, np.ndarray):
        arr = g

    arr = _add_diag(arr, mult_factor)
    arr = _normalize(arr)

    for i in range(max_loop):
        arr = _inflate(arr, inflate_factor)
        arr = _expand(arr, expand_factor)

        # Check for convergence
        if _stop(arr, i):
            break

    clusters = _get_clusters(arr)
    return arr, clusters
