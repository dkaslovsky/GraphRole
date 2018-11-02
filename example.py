import itertools as it

import networkx as nx
import numpy as np

from graphrole.features.recursive import RecursiveFeatureExtractor

np.random.seed(0)


def build_graph(n_nodes, directed=False):
    edge_generator = it.permutations if directed else it.combinations
    graph_constructor = nx.DiGraph if directed else nx.Graph

    all_edges = edge_generator(range(n_nodes), 2)
    edges = [edge for edge in all_edges if np.random.rand() > 0.75]
    
    return graph_constructor(edges)


if __name__ == '__main__':

    G = build_graph(20)

    rfe = RecursiveFeatureExtractor(G, max_generations=10)
    features = rfe.extract_features()
    print(features.T)
