import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
import matplotlib


def plot_graph(graph, ax=None, cmap='Spectral', **kwargs):
    """

    Parameters
    ----------
    graph : object
            A networkX or derived graph object

    ax : objext
         A MatPlotLib axes object

    cmap : str
           A MatPlotLib color map string. Default 'Spectral'

    Returns
    -------
    ax : object
         A MatPlotLib axes object. Either the argument passed in
         or a new object
    """
    if ax is None:
        ax = plt.gca()

    cmap = matplotlib.cm.get_cmap(cmap)

    # Setup edge color based on the health metric
    colors = []
    for s, d, e in graph.edges_iter(data=True):
        if hasattr(e, 'health'):
            colors.append(cmap(e.health)[0])
        else:
            colors.append(cmap(0)[0])

    nx.draw(graph, ax=ax, edge_color=colors)
    return ax


def plot_node(node, ax=None, clean_keys=[], **kwargs):
    """
    Plot the array and keypoints for a given node.

    Parameters
    ----------
    node : object
           A Node object from which data is extracted

    ax : object
         A MatPlotLIb axes object

    clean_keys : list
                 of strings of masking array names to apply

    kwargs : dict
             of MatPlotLib plotting options

    Returns
    -------
    ax : object
         A MatPlotLib axes object.  Either the argument passed in
         or a new object
    """

    if ax is None:
        ax = plt.gca()

    band = 1
    if 'band' in kwargs.keys():
        band = kwargs['band']
        kwargs.pop('band', None)

    array = node.get_array(band)

    ax.set_title(node.image_name)
    ax.margins(tight=True)
    ax.axis('off')

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'Greys'

    ax.imshow(array, cmap=cmap)

    keypoints = node.get_keypoints()
    if clean_keys:
        matches, mask = node._clean(clean_keys)
        keypoints = node.get_keypoints()[mask]

    marker = '.'
    if 'marker' in kwargs.keys():
        marker = kwargs['marker']
        kwargs.pop('marker', None)
    color = 'r'
    if 'color' in kwargs.keys():
        color = kwargs['color']
        kwargs.pop('color', None)
    ax.scatter(keypoints['x'], keypoints['y'], marker=marker, color=color, **kwargs)

    return ax


def plot_edge(edge, ax=None, clean_keys=[], image_space=100,
              scatter_kwargs={}, line_kwargs={}, image_kwargs={}):
    """
    Plot the correspondences for a given edge

    Parameters
    ----------
    edge : object
           A graph edge object

    ax : object
         A MatPlotLIb axes object

    clean_keys : list
                 of strings of masking array names to apply

    image_space : int
                  The number of pixels to insert between the images

    scatter_kwargs : dict
                     of MatPlotLib arguments to be applied to the scatter plots

    line_kwargs : dict
                  of MatPlotLib arguments to be applied to the lines connecting matches

    image_kwargs : dict
                   of MatPlotLib arguments to be applied to the image rendering

    Returns
    -------
    ax : object
         A MatPlotLib axes object.  Either the argument passed in
         or a new object
    """

    if ax is None:
        ax = plt.gca()

    # Plot setup
    ax.set_title('Matching: {} to {}'.format(edge.source.image_name,
                                             edge.destination.image_name))
    ax.margins(tight=True)
    ax.axis('off')

    # Image plotting
    source_array = edge.source.get_array()
    destination_array = edge.destination.get_array()

    s_shape = source_array.shape
    d_shape = destination_array.shape

    y = max(s_shape[0], d_shape[0])
    x = s_shape[1] + d_shape[1] + image_space
    composite = np.zeros((y, x))

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + image_space:] = destination_array

    if 'cmap' in image_kwargs:
        cmap = image_kwargs['cmap']
    else:
        cmap = 'Greys'

    ax.imshow(composite, cmap=cmap)

    matches, mask = edge._clean(clean_keys)

    source_keypoints = edge.source.get_keypoints(index=matches['source_idx'])
    destination_keypoints = edge.destination.get_keypoints(index=matches['destination_idx'])

    # Plot the source
    source_idx = matches['source_idx'].values
    s_kps = source_keypoints.loc[source_idx]
    ax.scatter(s_kps['x'], s_kps['y'], **scatter_kwargs, cmap='gray')

    # Plot the destination
    destination_idx = matches['destination_idx'].values
    d_kps = destination_keypoints.loc[destination_idx]
    x_offset = s_shape[1] + image_space
    newx = d_kps['x'] + x_offset
    ax.scatter(newx, d_kps['y'], **scatter_kwargs)

    # Draw the connecting lines
    color = 'y'
    if 'color' in line_kwargs.keys():
        color = line_kwargs['color']
        line_kwargs.pop('color', None)

    s_kps = s_kps[['x', 'y']].values
    d_kps = d_kps[['x', 'y']].values
    d_kps[:, 0] += x_offset

    for l in zip(s_kps, d_kps):
        ax.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]), color=color, **line_kwargs)

    return ax




