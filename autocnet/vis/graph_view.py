import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
import matplotlib

from skimage.transform import resize

def downsample(array, amount):
    return resize(array,
                      (int(array.shape[0] / amount),
                      int(array.shape[1] / amount)))

def plot_graph(graph, ax=None, cmap='Spectral', labels=False, font_size=12, clusters=None, **kwargs):
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

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax)
    if labels:
        labels = dict((i, d['image_name']) for i, d in graph.nodes_iter(data=True))
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size, ax=ax)
    ax.axis('off')
    return ax


def plot_node(node, ax=None, clean_keys=[], index_mask=None, downsampling=1, **kwargs):
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

    if isinstance(downsampling, bool):
        downsampling = node['downsample_amount']

    array = downsample(array, downsampling)

    ax.set_title(node['image_name'])
    ax.margins(tight=True)
    ax.axis('off')

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'Greys'

    ax.imshow(array, cmap=cmap)

    keypoints = node.get_keypoints(index=index_mask)

    if not keypoints.empty:
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


def plot_edge_decomposition(edge, ax=None, clean_keys=[], image_space=100,
                            scatter_kwargs={}, line_kwargs={}, image_kwargs={}):

    if ax is None:
        ax = plt.gca()

    # Plot setup
    ax.set_title('Matching: {} to {}'.format(edge.source['image_name'],
                                             edge.destination['image_name']))
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
    composite_decomp = np.zeros((y, x), dtype=np.int16)

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + image_space:] = destination_array

    composite_decomp[0: s_shape[0], :s_shape[1]] = edge.smembership
    composite_decomp[0: d_shape[0], s_shape[1] + image_space:] = edge.dmembership

    if 'cmap' in image_kwargs:
        cmap = image_kwargs['cmap']
    else:
        cmap = 'Greys'

    matches, mask = edge.clean(clean_keys)

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

    ax.imshow(composite, cmap=cmap)
    ax.imshow(composite_decomp, cmap='spectral', alpha=0.35)
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

def plot_edge(edge, ax=None, clean_keys=[], image_space=100, downsampling=1,
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

    downsample : bool

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
    ax.set_title('Matching: {} to {}'.format(edge.source['image_name'],
                                             edge.destination['image_name']))
    ax.margins(tight=True)
    ax.axis('off')

    # Image plotting
    if isinstance(downsampling, bool):
        downsample_source = edge.source['downsample_amount']
    else:
        downsample_source = downsampling
    source_array = edge.source.get_array()
    source_array = downsample(source_array, downsample_source)

    if isinstance(downsampling, bool):
        downsample_destin = edge.destination['downsample_amount']
    else:
        downsample_destin = downsampling
    destination_array = edge.destination.get_array()
    destination_array = downsample(destination_array, downsample_destin)

    s_shape = source_array.shape
    d_shape = destination_array.shape

    y = max(s_shape[0], d_shape[0])
    x = s_shape[1] + d_shape[1] + image_space
    composite = np.zeros((y, x))

    composite[0: s_shape[0], :s_shape[1]] = source_array
    composite[0: d_shape[0], s_shape[1] + image_space:] = destination_array

    if 'cmap' in image_kwargs:
        image_cmap = image_kwargs['cmap']
    else:
        image_cmap = 'Greys'

    matches, mask = edge.clean(clean_keys)

    if not matches.empty:
        source_keypoints = edge.source.get_keypoints(index=matches['source_idx'])
        destination_keypoints = edge.destination.get_keypoints(index=matches['destination_idx'])

        # Plot the source
        source_idx = matches['source_idx'].values
        s_kps = source_keypoints.loc[source_idx]
        ax.scatter(s_kps['x'], s_kps['y'], **scatter_kwargs)

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

    ax.imshow(composite, cmap=image_cmap)

    return ax

def cluster_plot(graph, ax=None, cmap='Spectral'):  # pragma: no cover
    """
    Parameters
    ----------
    graph : object
            A networkX or derived graph object

    ax : object
         A MatPlotLib axes object

    cmap : str
           A MatPlotLib color map string. Default 'Spectral'

    Returns
    -------
    ax : object
         A MatPlotLib axes object that was either passed in
         or a new axes object
    """
    if ax is None:
        ax = plt.gca()

    if not hasattr(graph, 'clusters'):
        raise AttributeError('Clusters have not been computed.')

    cmap = matplotlib.cm.get_cmap(cmap)

    colors = []

    for i, n in graph.nodes_iter(data=True):
        for j in enumerate(graph.clusters):
            if i in graph.clusters.get(j[1]):
                colors.append(cmap(j[1])[0])
                continue

    nx.draw(graph, ax=ax, node_color=colors)
    return ax

def plot_matches(matches, a, b, extent=200):
    nsubplots = len(matches)
    figs = []
    for i, (idx, row) in enumerate(matches.iterrows()):
        fig, axes = plt.subplots(1, 2)
        sx = row.source_x
        sy = row.source_y
        dx = row.destination_x
        dy = row.destination_y

        suba = a.read_array(pixels=[int(sx-extent), int(sy-extent), int(2*extent), int(2*extent)])
        subb = b.read_array(pixels=[int(dx-extent), int(dy-extent), int(2*extent), int(2*extent)])

        axes[0].imshow(suba, cmap='Greys')
        axes[1].imshow(subb, cmap='Greys')
        figs.append(fig)

    return figs
