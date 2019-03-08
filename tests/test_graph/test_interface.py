import itertools as it
import unittest

import networkx as nx
import pandas as pd

from graphrole.graph import interface

# try to import igraph; if not found set flag to skip associated tests
skip_igraph_tests = False
try:
    import igraph as ig
except ImportError:
    skip_igraph_tests = True


class TestGetInterface(unittest.TestCase):

    """ Unit tests for get_interface() """

    def test_get_interface(self):
        # test with object from supported library
        obj = nx.Graph()
        graph_interface = interface.get_interface(obj)
        self.assertIsInstance(graph_interface, interface.BaseGraphInterface)
        # test with object not from supported library
        obj = str
        graph_interface = interface.get_interface(obj)
        self.assertIsNone(graph_interface)
        # test with object with no module property
        obj = 'str'
        graph_interface = interface.get_interface(obj)
        self.assertIsNone(graph_interface)


class BaseGraphInterfaceTest:

    class BaseGraphInterfaceTestCases(unittest.TestCase):

        """ Unit tests for interfaces to graph libraries """

        graph = None
        graph_empty = None
        graph_directed_weighted = None

        nodes = range(7)
        edges = [
            (0, 1), (0, 2), (0, 3),
            (3, 6), (4, 5), (4, 6),
            (5, 6)
        ]
        weights = [
            2, 1.5, 3,
            0.25, 0.75, 2.5,
            1
        ]

        def test_get_num_edges(self):
            self.assertEqual(self.graph.get_num_edges(), len(self.edges))
            self.assertEqual(self.graph_empty.get_num_edges(), 0)

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
            pd.testing.assert_frame_equal(result_features, expected_features)

        def test_get_neighborhood_features_with_directed_weighted_graph(self):
            features = [
                pd.Series(
                    {
                        0: 0.00, 1: 2.00, 2: 1.50, 3: 3.00,
                        4: 0.00, 5: 0.75, 6: 3.75
                    }
                ).rename('in_degree'),
                pd.Series(
                    {
                        0: 6.50, 1: 0.00, 2: 0.00, 3: 0.25,
                        4: 3.25, 5: 1.00, 6: 0.00
                    }
                ).rename('out_degree'),
                pd.Series(
                    {
                        0: 6.50, 1: 2.00, 2: 1.50, 3: 3.25,
                        4: 3.25, 5: 1.75, 6: 3.75
                    }
                ).rename('total_degree'),                
                pd.Series(
                    {
                        0: 6.50, 1: 0.00, 2: 0.00, 3: 0.25,
                        4: 4.25, 5: 1.00, 6: 0.00
                    }
                ).rename('internal_edges'),
                pd.Series(
                    {
                        0: 0.25, 1: 0.00, 2: 0.00, 3: 0.00,
                        4: 0.00, 5: 0.00, 6: 0.00
                    }
                ).rename('external_edges')
            ]
            expected_features = pd.concat(features, axis=1)
            result_features = self.graph_directed_weighted.get_neighborhood_features()
            pd.testing.assert_frame_equal(result_features, expected_features)


class TestNetworkxInterface(BaseGraphInterfaceTest.BaseGraphInterfaceTestCases):

    """ Unit tests for Networkx interface """

    @classmethod
    def setUpClass(cls):
        G_empty = nx.Graph()
        cls.graph_empty = interface.NetworkxInterface(G_empty)

        G = nx.Graph(cls.edges)
        cls.graph = interface.NetworkxInterface(G)

        G_directed_weighted = nx.DiGraph()
        for edge, weight in zip(cls.edges, cls.weights):
            G_directed_weighted.add_edge(*edge, weight=weight)
        cls.graph_directed_weighted = interface.NetworkxInterface(G_directed_weighted)


class TestIgraphInterface(BaseGraphInterfaceTest.BaseGraphInterfaceTestCases):

    """ Unit tests for Igraph interface """

    @unittest.skipIf(
        skip_igraph_tests,
        reason='igraph not found, skipping associated interface tests'
    )
    @classmethod
    def setUpClass(cls):
        G_empty = ig.Graph()
        cls.graph_empty = interface.IgraphInterface(G_empty)

        G = ig.Graph()
        G.add_vertices(len(cls.nodes))
        G.add_edges(cls.edges)
        cls.graph = interface.IgraphInterface(G)

        G_directed_weighted = ig.Graph(directed=True)
        G_directed_weighted.add_vertices(len(cls.nodes))
        for edge, weight in zip(cls.edges, cls.weights):
            G_directed_weighted.add_edge(*edge, weight=weight)
        cls.graph_directed_weighted = interface.IgraphInterface(G_directed_weighted)
