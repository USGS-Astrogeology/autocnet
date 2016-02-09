import math
import numpy as np

from autocnet.examples import get_path
from autocnet.fileio.io_gdal import GeoDataset
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as mplPolygon


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

    keypoints = node.keypoints
    if clean_keys:
        mask = np.prod([node._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
        keypoints = node.keypoints[mask]

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
    composite = np.zeros((y,x))

    composite[:, :s_shape[1]] = source_array
    composite[:, s_shape[1] + image_space:] = destination_array

    if 'cmap' in image_kwargs:
        cmap = image_kwargs['cmap']
    else:
        cmap = 'Greys'

    ax.imshow(composite, cmap=cmap)

    # Match point plotting
    source_keypoints = edge.source.keypoints
    destination_keypoints = edge.destination.keypoints

    matches = edge.matches

    if clean_keys:
        mask = np.prod([edge._mask_arrays[i] for i in clean_keys], axis=0, dtype=np.bool)
        matches = edge.matches[mask]

    marker = '.'
    if 'marker' in scatter_kwargs.keys():
        marker = scatter_kwargs['marker']
        scatter_kwargs.pop('marker', None)

    color = 'r'
    if 'color' in scatter_kwargs.keys():
        color = scatter_kwargs['color']
        scatter_kwargs.pop('color', None)

    # Plot the source
    source_idx = matches['source_idx'].values
    s_kps = source_keypoints.iloc[source_idx]
    ax.scatter(s_kps['x'], s_kps['y'], marker=marker, color=color, **scatter_kwargs)

    # Plot the destination
    destination_idx = matches['destination_idx'].values
    d_kps = destination_keypoints.iloc[destination_idx]
    x_offset = s_shape[0] + image_space
    newx = d_kps['x'] + x_offset
    ax.scatter(newx, d_kps['y'], marker=marker, color=color, **scatter_kwargs)

    # Draw the connecting lines
    color = 'y'
    if 'color' in line_kwargs.keys():
        color = line_kwargs['color']
        line_kwargs.pop('color', None)

    s_kps = s_kps[['x', 'y']].values
    d_kps = d_kps[['x', 'y']].values
    d_kps[:,0] += x_offset

    for l in zip(s_kps, d_kps):
        ax.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]), color=color, **line_kwargs)

    return ax

def plotAdjacencyGraphFeatures(graph, pointColorAndHatch='b.', featurePointSize=7):
    """
    Plot each image in an adjacency graph and its found features in a single figure.
    The user may also specify the color and style of the points to be plotted and
    the size of the points.

    Parameters
    ----------
    graph : object
            A CandicateGraph object from a JSON file whose node (i.e. image)
            keypoints have been filled using the autocnet.feature_extractor.

    pointColorAndHatch : str
                         The color and hatch (symbol) to be used to mark
                         the found features. Defaults to 'b.', blue and 
                         square dot. See matplotlib documentation for
                         more choices.

    featurePointSize : int
                       The size of the point marker. Defaults to 7.

    """

    counter = 1
    columns = math.ceil(math.sqrt(graph.number_of_nodes()))
    rows = columns
    for node, attributes in graph.nodes_iter(data=True):
        plt.subplot(rows, columns, counter)
        plotFeatures(attributes['handle'], 
                         attributes['keypoints'], 
                         pointColorAndHatch)
        counter = counter + 1




# NOTE: We will probably delete this code if it is found to be un-needed,
# However, for now we will keep in case it winds up being a more useful tool.
def plotAdjacencyGraphMatchesSingleDisplay(imageName1, 
                                           imageName2, 
                                           graph,
                                           featurePointSize=10,
                                           lineWidth=3):
    """
    This is an earlier version of plotAdjacencyGraphMatches() where the
    images are offset in a single display box rather than in their
    own subplots.

    Parameters
    ----------
    imageName : str
                The name of the first image file (with extension, without path).
                This will be the title of the left subplot.

    imageName : str
                The name of the second image file (with extension, without path).
                This will be the title of the right subplot.

    graph : object
            A CandicateGraph object containing the given images (as nodes) and their
            matches (edges). This graph is read from a JSON file, autocnet.feature_extractor
            has been applied, and FlannMatcher has been applied.

    featurePointSize : int
                       The size of the feature point marker. Defaults to 10.

    lineWidth : int
                The width of the match lines. Defaults to 3.

    Returns
    -------
     : AxesImage object
       An image object that can be saved. 
    """

    imgArray1 = GeoDataset(get_path(imageName1)).read_array()
    imgArray2 = GeoDataset(get_path(imageName2)).read_array()
    
    height1, width1 = imgArray1.shape[:2]
    height2, width2 = imgArray2.shape[:2]
    
    w = width1+width2+50
    h = max(height1, height2)
    
    displayBox = np.zeros((h, w), np.uint8)
       
    displayBox[:height1, :width1] = imgArray1
    displayBox[:height2, width1+50:w] = imgArray2
    
    for kp in graph.get_keypoints(imageName1): 
        x, y = kp.pt
        plt.plot(x, y,'ro', markersize=featurePointSize)
    for kp in graph.get_keypoints(imageName2): 
        x, y = kp.pt
        plt.plot(x+width1+50, y,'ro', markersize=featurePointSize)
    
    edge = graph[graph.node_name_map[imageName1]][graph.node_name_map[imageName2]]
    if 'matches' in edge.keys():
        for i, row in edge['matches'].iterrows():
            # get matching points
            image1ID = int(row['source_idx'])
            image2ID = int(row['destination_idx'])
            keypointImage1 = (graph.get_keypoints(imageName1)[image1ID].pt[0],
                              graph.get_keypoints(imageName1)[image1ID].pt[1])
            keypointImage2 = (graph.get_keypoints(imageName2)[image2ID].pt[0]+width1+50,
                              graph.get_keypoints(imageName2)[image2ID].pt[1])
                            
            # construct a line between the matching points using the data coordinates and the 
            # transformation from data coordinates to display coordinates
            plt.plot([keypointImage1[0], keypointImage2[0]], 
                     [keypointImage1[1], keypointImage2[1]],
                     color='g', 
                     marker='o',
                     markeredgecolor='g',
                     markersize=featurePointSize,
                     linewidth=lineWidth,
                     alpha=0.5)
    return plt.imshow(displayBox, cmap='Greys')
