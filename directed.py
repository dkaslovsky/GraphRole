import argparse
import random

import igraph as ig
import networkx as nx
import numpy as np

from graphrole.graph.interface import  IgraphInterface, NetworkxInterface


random.seed(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directed', action='store_true')
    parser.add_argument('-w', '--weighted', action='store_true')
    args = parser.parse_args()

    n_nodes = 10
    n_edges = 25

    nodes = range(n_nodes)
    edges = set()
    while len(edges) < n_edges:
        from_node = random.choice(nodes)
        to_node = random.choice(nodes)
        if from_node == to_node:
            continue
        if args.directed:
            edge = (from_node, to_node)
        else:
            edge = tuple(sorted((from_node, to_node)))
        edges.add(edge)
    edges = list(edges)

    weights = [round(10 * random.random(), 1) for _ in range(n_edges)]

    # networkx
    nG = nx.DiGraph() if args.directed else nx.Graph()
    nG.add_nodes_from(nodes)
    if args.weighted:
        for edge, weight in zip(edges, weights):
            nG.add_edge(*edge, weight=weight)
    else:
        nG.add_edges_from(edges)
    nG_interface = NetworkxInterface(nG)
    nG_nbrhd_features = nG_interface.get_neighborhood_features()
    print('\nnetworkx')
    print(nG_nbrhd_features)

    # igraph
    iG = ig.Graph(directed=args.directed)
    iG.add_vertices(n_nodes)    
    if args.weighted:
        for edge, weight in zip(edges, weights):
            iG.add_edge(*edge, weight=weight)
    else:
        iG.add_edges(edges)
    iG_interface = IgraphInterface(iG)
    iG_nbrhd_features = iG_interface.get_neighborhood_features()
    print('\nigraph')
    print(iG_nbrhd_features)

    if not np.allclose(nG_nbrhd_features.values, iG_nbrhd_features):
        print('Mismatch:')
        print(nG_nbrhd_features - iG_nbrhd_features)
