from abc import ABC, abstractmethod

import networkx as nx
import pandas as pd

# TODO: directed and weighted versions needed



# Abstract class not needed but used to define the interface for
# classes that wrap graph libraries like networkx
class Graph(ABC):

    @abstractmethod
    def get_neighborhood_features(self):
        pass
    
    @abstractmethod
    def get_nodes(self):
        pass

    @abstractmethod
    def get_neighbors(self):
        pass


class NetworkxGraph(Graph):

    def __init__(self, G):
        self.G = G
    
    def get_neighborhood_features(self):
        local = self._get_local_features()
        ego = self._get_egonet_features()
        return (pd.concat([local, ego], axis=1)
                .rename_axis('node', axis=0)
                .sort_index())

    def get_nodes(self):
        return self.G.nodes

    def get_neighbors(self, node):
        return self.G[node].keys()

    def _get_local_features(self):
        return pd.DataFrame.from_dict(
            dict(self.G.degree),
            orient='index',
            columns=['degree']
        )

    def _get_egonet_features(self):
        egonet_features = {}
        for node in self.G.nodes:
            ego = nx.ego_graph(self.G, node, radius=1)
            features = {
                'n_internal_edges': len(ego.edges),
                'n_external_edges': len(list(nx.edge_boundary(self.G, ego.nodes)))
            }
            egonet_features[node] = features        
        return pd.DataFrame.from_dict(egonet_features, orient='index')
