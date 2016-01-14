from collections import Hashable
import networkx as nx

from autocnet.control.control import C
from autocnet.fileio import io_json



class CandidateGraph(nx.DiGraph):
    """
    A NetworkX derived directed graph to store candidate overlap images.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self,*args, **kwargs):
        super(CandidateGraph, self).__init__(*args, **kwargs)

    def add_image(self, identifier, *args, **kwargs):
        """
        Parameters
        ==========
        identifier : object
                     A Python hashable object to be used as the node key
        """
        if isinstance(identifier, Hashable):
            self.add_node(identifier, *args, **kwargs)
        else:
            raise TypeError('{} is not hashable and can not be a node id'.format(identifier))

    def adjacency_to_json(self, outputfile):
        """
        Write the edge structure to a JSON adjacency list

        Parameters
        ==========

        outputfile : str
                     PATH where the JSON will be written
        """
        adjacency_dict = {}
        for n in self.nodes():
            adjacency_dict[n] = self.neighbors(n)
        io_json.write_json(adjacency_dict, outputfile)

    def cnet_from_graph(self):
        """
        Create a control network from a graph or subgraph

        Returns
        -------
        cnet : object
               A control network object
        """
        pass

    @classmethod
    def from_adjacency(cls, inputfile):
        """
        Instantiate the class using an adjacency list

        Parameters
        ----------
        inputfile : str
                    The input file containing the graph representation

        Returns
        -------
         : object
           A Network graph object

        Examples
        --------
        >>> from autocnet.examples import get_path
        >>> inputfile = get_path('adjacency.json')
        >>> candidate_graph = network.CandidateGraph.from_adjacency(inputfile)
        """
        adjacency_dict = io_json.read_json(inputfile)
        return cls(adjacency_dict)

