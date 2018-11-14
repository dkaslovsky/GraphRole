from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, TypeVar, Union

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


# TODO: CHANGE NAME!
# TODO: can't rely on .index, not the same as identifier, use 'name'
# TODO: add a check_names method to add names matching index if G doesn't have names
# TODO: docstring(s) for _get_edge_boundary
# TODO: add example with igraph
# TODO: return type for _get_edge_boundary isn't right
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
        return (vertex.index for vertex in self.G.vs())

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
                'external_edges': len(self._get_edge_boundary(self.G, ego))
            }
            egonet_features[node] = features
        return pd.DataFrame.from_dict(egonet_features, orient='index')

    @staticmethod
    def _get_edge_boundary(graph: ig.Graph, subgraph: ig.Graph) -> List[Edge]:

        def _is_boundary(edge, vertex_seq1, vertex_seq2):
            vertices1 = set(vertex_seq1['name'])
            vertices2 = set(vertex_seq2['name'])
            e1, e2 = edge.tuple
            return (
                (e1 in vertices1 and e2 in vertices2)
                or
                (e1 in vertices2 and e2 in vertices1)
            )

        sub_vertices = subgraph.vs()
        sub_vertices_comp = graph.vs().select(lambda x: x['name'] not in sub_vertices['name'])
        return graph.es().select(lambda x: _is_boundary(x, sub_vertices, sub_vertices_comp))
