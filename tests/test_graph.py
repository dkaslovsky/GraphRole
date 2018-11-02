import itertools as it
import unittest

import networkx as nx
import pandas as pd

from graphrole.graph import NetworkxGraph


class TestNetworkxGraph(unittest.TestCase):

    """ Unit tests for NetworkxGraph """

    edges = [
        (0, 1), (0, 2), (0, 3),
        (3, 6), (4, 5), (4, 6),
        (5, 6)
    ]
    
    graph = NetworkxGraph(nx.Graph(edges))
        
    def test_get_nodes(self):
        nodes = self.graph.get_nodes()
        expected_nodes = set(it.chain.from_iterable(self.edges))
        self.assertSetEqual(set(nodes), expected_nodes)

    def test_get_neighbors(self):
        table = {
            'node 0': {
                'node': 0,
                'nbrs': {1, 2, 3}
            },
            'node 1': {
                'node': 1,
                'nbrs': {0}
            },
            'node 2': {
                'node': 2,
                'nbrs': {0}
            },
            'node 3': {
                'node': 3,
                'nbrs': {0, 6}
            },
            'node 4': {
                'node': 4,
                'nbrs': {5, 6}
            },
            'node 5': {
                'node': 5,
                'nbrs': {4, 6}
            },
            'node 6': {
                'node': 6,
                'nbrs': {3, 4, 5}
            },
        }
        for test_name, test in table.items():
            nbrs = set(self.graph.get_neighbors(test['node']))
            self.assertSetEqual(nbrs, test['nbrs'], test_name)

    def test_get_neighborhood_features(self):
        features = [
            pd.Series(
                {
                    0: 3, 1: 1, 2: 1, 3: 2,
                    4: 2, 5: 2, 6: 3
                }
            ).rename('degree'),
            pd.Series(
                {
                    0: 3, 1: 1, 2: 1, 3: 2,
                    4: 3, 5: 3, 6: 4
                }
            ).rename('internal_edges'),
            pd.Series(
                {
                    0: 1, 1: 2, 2: 2, 3: 4,
                    4: 1, 5: 1, 6: 1
                }
            ).rename('external_edges')
        ]
        expected_features = pd.concat(features, axis=1)
        result_features = self.graph.get_neighborhood_features()
        self.assertTrue(result_features.equals(expected_features))
