from collections import defaultdict
from numbers import Number
from typing import Iterable

import networkx as nx
import pandas as pd

from graphrole.graph.interface import BaseGraphInterface
from graphrole.types import Edge, Node


class NetworkxInterface(BaseGraphInterface):

    """ Interface for Networkx Graph object """

    def __init__(self, G: nx.Graph, **kwargs) -> None:
        """
        :param G: Networkx Graph
        :kwarg attributes: boolean indicating whether to use node attributes as features
        :kwarg attributes_include: include list of node attributes for features
          (all attributes are used if not specified)
        :kwarg attributes_exclude: exclude list of node attributes for features
          (overrides attributes_include in cases of conflict)
        """
        self.G = G
        self.directed = G.is_directed()

        self._set_attribute_kwargs(**kwargs)

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
            features = pd.DataFrame({
                'in_degree': dict(self.G.in_degree(weight='weight')),
                'out_degree': dict(self.G.out_degree(weight='weight')),
                'total_degree': dict(self.G.degree(weight='weight')),
            })
        else:
            features = pd.DataFrame.from_dict(
                dict(self.G.degree(weight='weight')),
                orient='index',
                columns=['degree'],
            )
        
        if self._attrs:
            attribute_features = self._get_attribute_features()
            features = pd.concat([features, attribute_features], axis=1)

        return features.fillna(0)

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

    def _get_attribute_features(self) -> pd.DataFrame:
        """
        Return attribute features for each node in the graph
        """
        exclude = {item for item in self._attrs_exclude}

        if self._attrs_include:
            attrs = {
                self._attribute_feature_name(attr_name): {
                    node_idx: attr_val
                    for node_idx, attrs in self.G.nodes(data=True)
                    if isinstance(attr_val := attrs.get(attr_name, 0), Number)
                }
                for attr_name in self._attrs_include
                if attr_name not in exclude
            }
            return pd.DataFrame(attrs).fillna(0)

        attrs = defaultdict(dict)
        for node_idx, node_attrs in self.G.nodes(data=True):
            for attr_name, attr_val in node_attrs.items():
                if attr_name in exclude:
                    continue
                if not isinstance(attr_val, Number):
                    continue
                attrs[self._attribute_feature_name(attr_name)][node_idx] = attr_val
        return pd.DataFrame(attrs).fillna(0)

    def _get_edge_sum(self, edges: Iterable[Edge]) -> float:
        """
        Return weighted sum of edges (total number of edges if unweighted)
        :param edges: edges to sum
        """
        return sum(
            self.G.get_edge_data(*edge, default={}).get('weight', 1)
            for edge in edges
        )
