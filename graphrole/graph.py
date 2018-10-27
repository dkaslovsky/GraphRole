import itertools as it

import networkx as nx
import numpy as np
import pandas as pd

from graphrole.features import group_features, vertical_log_binning

np.random.seed(0)


# TODO: directed, weighted

def build_graph(n_nodes, directed=False):
    edge_generator = it.permutations if directed else it.combinations
    graph_constructor = nx.DiGraph if directed else nx.Graph
    all_edges = edge_generator(range(n_nodes), 2)
    edges = [edge for edge in all_edges if np.random.rand() > 0.75]
    return graph_constructor(edges)


def get_local_features(G: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        dict(G.degree),
        orient='index',
        columns=['degree']
    )


def get_egonet_features(G: nx.Graph) -> pd.DataFrame:
    egonet_features = {}
    for node in G.nodes:
        ego = nx.ego_graph(G, node, radius=1)
        features = {
            'n_internal_edges': len(ego.edges),
            'n_external_edges': len(list(nx.edge_boundary(G, ego.nodes)))
        }
        egonet_features[node] = features        
    return pd.DataFrame.from_dict(egonet_features, orient='index')


def get_neighborhood_features(G: nx.Graph) -> pd.DataFrame:
    local = get_local_features(G)
    ego = get_egonet_features(G)
    df = pd.concat([local, ego], axis=1)
    df.index.rename('node', inplace=True)
    return df.sort_index()


class RecursiveFeatureExtractor:

    recursive_aggs = [
        pd.DataFrame.sum,
        pd.DataFrame.mean,
    ]

    def __init__(self,
                 G: nx.Graph,
                 max_levels: int = 10):
        
        self.G = G
        self.generation_count = 0
        self.features = get_neighborhood_features(self.G)
        self.binned_features = self.features.apply(vertical_log_binning, axis=0)
        self.generations = {column: self.generation_count for column in self.features.columns}
    
    # TODO: node type for typing?
    def get_neighbors(self, node):
        return self.G[node].keys()
    
    def recursive_round(self):
        rec_features = {}
        for node in self.G.nodes:
            nbrs = self.get_neighbors(node)
            nbr_features = self.features.loc[nbrs]
            rec_nbr_features = nbr_features.agg(self.recursive_aggs)

            features = {}
            for row in rec_nbr_features.reset_index().to_dict(orient='records'):
                agg = row.pop('index')
                row_features = {'{}_{}'.format(key, agg): val for key, val in row.items()}
                # TODO: Python 3 style dict update
                features.update(row_features)
            rec_features[node] = features
        return pd.DataFrame.from_dict(rec_features, orient='index')
    
    def add_features(self, features: pd.DataFrame):
        self.generation_count += 1
        col_names = {col: '{}_gen{}'.format(col, self.generation_count)
                     for col in features.columns}
        self.features = pd.concat([self.features, features], axis=1).rename(columns=col_names)
        binned_features = features.apply(vertical_log_binning, axis=0)
        self.binned_features = pd.concat([self.binned_features, binned_features], axis=1).rename(columns=col_names)
        # TODO: Python 3 style dict update
        self.generations.update({feature_name: self.generation_count
                                 for feature_name in col_names.values()})
        return
    
    def prune_features(self, features):
        pass
            
    

            

if __name__ == '__main__':

    G = build_graph(10)
    
    rfe = RecursiveFeatureExtractor(G)
    new_features = rfe.recursive_round()
    rfe.add_features(new_features)
    new_features = rfe.recursive_round()
    rfe.add_features(new_features)
