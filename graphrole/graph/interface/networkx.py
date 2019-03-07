from typing import Iterable

import networkx as nx
import pandas as pd

from graphrole.graph.interface import BaseGraphInterface
from graphrole.types import Edge, Node


class NetworkxInterface(BaseGraphInterface):

    """ Interface for Networkx Graph object """

    def __init__(self, G: nx.Graph) -> None:
        """
        :param G: Networkx Graph
        """
        self.G = G
        self.directed = G.is_directed()

    def get_num_edges(self) -> int:
        """
        Return number of edges in the graph
        """
        return self.G.number_of_edges()

    def get_nodes(self) -> Iterable[Node]:
        """
        Return iterable of nodes in the graph
        """
        return self.G.nodes

    def get_neighbors(self, node: Node) -> Iterable[Node]:
        """
        Return iterable of neighbors of specified node
        """
        return self.G[node].keys()

    def _get_local_features(self) -> pd.DataFrame:
        """
        Return local features for each node in the graph
        """
        if self.directed:
            return pd.DataFrame(
                {
                    'in_degree': dict(self.G.in_degree(weight='weight')),
                    'out_degree': dict(self.G.out_degree(weight='weight')),
                    'total_degree': dict(self.G.degree(weight='weight')),
                }
            )

        return pd.DataFrame.from_dict(
            dict(self.G.degree(weight='weight')),
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
            ego_boundary = list(nx.edge_boundary(self.G, ego.nodes))
            egonet_features[node] = {
                'internal_edges': self._get_edge_sum(ego.edges),
                'external_edges': self._get_edge_sum(ego_boundary)
            }
        return pd.DataFrame.from_dict(egonet_features, orient='index')

    ### helpers ###

    def _get_edge_sum(self, edges: Iterable[Edge]) -> float:
        """
        Return weighted sum of edges (total number of edges if unweighted)
        :param edges: edges to sum
        """
        return sum(
            self.G.get_edge_data(*edge, default={})\
             .get('weight', 1)
            for edge in edges
        )
