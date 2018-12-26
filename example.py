import itertools as it

import igraph as ig
import networkx as nx
import numpy as np

from graphrole.features.extract import RecursiveFeatureExtractor

np.random.seed(0)


def get_edges(n_nodes, directed=False):
    edge_generator = it.permutations if directed else it.combinations
    all_edges = edge_generator(range(n_nodes), 2)
    edges = [edge for edge in all_edges if np.random.rand() > 0.75]
    return edges


def build_networkx_graph(edges, directed=False):
    graph_constructor = nx.DiGraph if directed else nx.Graph    
    return graph_constructor(edges)


def build_igraph_graph(edges, directed=False):
    # TODO: directed?
    n_nodes = 1 + max(it.chain.from_iterable(edges))
    graph = ig.Graph(n_nodes)
    graph.add_edges(edges)
    return graph


if __name__ == '__main__':

    edges = get_edges(20)

    nG = build_networkx_graph(edges)
    iG = build_igraph_graph(edges)

    rfe = RecursiveFeatureExtractor(nG, max_generations=10)
    features = rfe.extract_features()
    print('networkx')
    print(features.T)

    rfe = RecursiveFeatureExtractor(iG, max_generations=10)
    features = rfe.extract_features()
    print('igraph')
    print(features.T)
