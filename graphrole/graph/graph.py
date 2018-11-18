from abc import ABC, abstractmethod
from functools import partial
from typing import Iterable, List, Set, Tuple, TypeVar, Union

import igraph as ig
import networkx as nx
import pandas as pd


NodeName = Union[int, str]
Edge = Tuple[NodeName, NodeName]
GraphInterface = TypeVar('GraphInterface', bound='Graph')


class Graph(ABC):

    """
    Abstract class to define the interface used to interact with various graph libraries
    """

    @abstractmethod
    def get_neighborhood_features(self) -> pd.DataFrame:
        """
        Return neighborhood features (local + egonet) for each node in the graph
        """
        pass
    
    @abstractmethod
    def get_nodes(self) -> Iterable[NodeName]:
        """
        Return iterable of nodes in the graph
        """
        pass

    @abstractmethod
    def get_neighbors(self, node: NodeName) -> Iterable[NodeName]:
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

    def get_nodes(self) -> Iterable[NodeName]:
        """
        Return iterable of nodes in the graph
        """
        return self.G.nodes

    def get_neighbors(self, node: NodeName) -> Iterable[NodeName]:
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


class IgraphGraph(Graph):

    """ Interface for igraph Graph object """

    def __init__(self, G: ig.Graph) -> None:
        """
        :param G: igraph Graph
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

    def get_nodes(self) -> Iterable[NodeName]:
        """
        Return iterable of nodes in the graph
        """
        return self.G.vs().indices

    def get_neighbors(self, node: NodeName) -> Iterable[NodeName]:
        """
        Return iterable of neighbors of specified node
        """
        return self.G.neighbors(node)

    def _get_local_features(self) -> pd.DataFrame:
        """
        Return local features for each node in the graph
        """
        degree_dict = {vertex.index: vertex.degree() for vertex in self.G.vs()}
        return pd.DataFrame.from_dict(
            degree_dict,
            orient='index',
            columns=['degree']
        )

    def _get_egonet_features(self) -> pd.DataFrame:
        """
        Return egonet features for each node in the graph
        """
        egonet_features = {}
        for node in self.get_nodes():
            ego_nodes = self.G.neighborhood(node, order=1)
            ego = self.G.induced_subgraph(ego_nodes)
            features = {
                'internal_edges': len(ego.es()),
                'external_edges': len(self._get_edge_boundary(ego_nodes))
            }
            egonet_features[node] = features
        return pd.DataFrame.from_dict(egonet_features, orient='index')

    def _get_edge_boundary(self, interior_vertex_ids: List[NodeName]) -> List[Edge]:
        """
        Return the list of edges on the boundary of the vertex sets defined
        by interior_vertex_ids and its complement
        :param interior_vertex_ids: list of vertex ids defining the interior
        """
        interior = set(interior_vertex_ids)
        exterior = set(self.get_nodes()) - interior
        _is_boundary = partial(self._is_boundary, interior=interior, exterior=exterior)
        return [edge.tuple for edge in self.G.es() if _is_boundary(edge.tuple)]

    @staticmethod
    def _is_boundary(edge: Edge, interior: Set[NodeName], exterior: Set[NodeName]) -> bool:
        """
        Return True if edge is on the boundary of the interior and exterior vertex sets (else False)
        :param edge: edge to evaluate
        :param interior: set of vertex ids
        :param exterior: set of vertex ids
        """
        v1, v2 = edge
        return (
            (v1 in interior and v2 in exterior)
            or
            (v1 in exterior and v2 in interior)
        )
