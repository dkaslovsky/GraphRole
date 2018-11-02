from abc import ABC, abstractmethod
from typing import Any, Iterable

import networkx as nx
import pandas as pd


class Graph(ABC):

    """
    Abstract class to define the interface used in graphrole.features.recursive 
    module to interact with various graph libraries
    """

    @abstractmethod
    def get_neighborhood_features(self) -> pd.DataFrame:
        """
        Return neighborhood features (local + egonet) for each node in the graph
        """
        pass
    
    @abstractmethod
    def get_nodes(self) -> Iterable:
        """
        Return iterable of nodes in the graph
        """
        pass

    @abstractmethod
    def get_neighbors(self, node: Any) -> Iterable:
        """
        Return iterable of neighbors of specified node
        """
        pass


class NetworkxGraph(Graph):

    """ Interface for Networkx Graph object """

    def __init__(self, G: nx.Graph) -> None:
        """
        :param G: Networkx Graph
        """
        self.G = G
    
    def get_neighborhood_features(self) -> pd.DataFrame:
        """
        Return neighborhood features (local + egonet) for each node in the graph
        """
        local = self._get_local_features()
        ego = self._get_egonet_features()
        return (pd.concat([local, ego], axis=1)
                .rename_axis('node', axis=0)
                .sort_index())

    def get_nodes(self) -> Iterable:
        """
        Return iterable of nodes in the graph
        """
        return self.G.nodes

    def get_neighbors(self, node: Any) -> Iterable:
        """
        Return iterable of neighbors of specified node
        """
        return self.G[node].keys()

    def _get_local_features(self) -> pd.DataFrame:
        """
        Return local features for each node in the graph
        """
        return pd.DataFrame.from_dict(
            dict(self.G.degree),
            orient='index',
            columns=['degree']
        )

    def _get_egonet_features(self) -> pd.DataFrame:
        """
        Return egonet features for each node in the graph
        """
        egonet_features = {}
        for node in self.G.nodes:
            ego = nx.ego_graph(self.G, node, radius=1)
            features = {
                'internal_edges': len(ego.edges),
                'external_edges': len(list(nx.edge_boundary(self.G, ego.nodes)))
            }
            egonet_features[node] = features        
        return pd.DataFrame.from_dict(egonet_features, orient='index')
