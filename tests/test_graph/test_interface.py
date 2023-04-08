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


def assertion_error_with_test_case(err: AssertionError, test_case: str) -> AssertionError:
    """ Wrap assertion error with a test case string """
    msg = f'\ntest case: {test_case}\n{err.args[0]}'
    err.args = (msg,)
    return err

class TestGetInterface(unittest.TestCase):

    """ Unit tests for get_interface() """

    def test_get_interface(self):
        # test with object from supported library
        obj = nx.Graph()
        graph_interface = interface.get_interface(obj)
        graph = graph_interface(obj)
        self.assertIsInstance(graph, interface.BaseGraphInterface)
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
        attrs = {
            0: {'attr1': 1.00, 'attr2': 0.00},
            1: {'attr2': 1.00},
            2: {'attr2': 2.00},
            3: {'attr2': 3.00},
            4: {'attr2': 4.00},
            5: {'attr2': 5.00},
            6: {'attr2': 6.00},
        }

        def test_get_num_edges(self):
            test_graph = self.klass(self.graph)
            test_graph_empty = self.klass(self.graph_empty)
            self.assertEqual(test_graph.get_num_edges(), len(self.edges))
            self.assertEqual(test_graph_empty.get_num_edges(), 0)

        def test_get_nodes(self):
            test_graph = self.klass(self.graph)
            nodes = test_graph.get_nodes()
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

            test_graph = self.klass(self.graph)

            for test_name, test in table.items():    
                nbrs = set(test_graph.get_neighbors(test['node']))
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
            test_graph = self.klass(self.graph)
            expected_features = pd.concat(features, axis=1)
            result_features = test_graph.get_neighborhood_features()
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
            test_graph = self.klass(self.graph_directed_weighted)
            expected_features = pd.concat(features, axis=1)
            result_features = test_graph.get_neighborhood_features()
            pd.testing.assert_frame_equal(result_features, expected_features)

        def test_get_neighborhood_features_with_attributes(self):
            features = [
                pd.Series(
                    {
                        0: 3, 1: 1, 2: 1, 3: 2,
                        4: 2, 5: 2, 6: 3
                    }
                ).rename('degree'),
                pd.Series(
                    {
                        0: 1.00, 1: 0.00, 2: 0.00, 3: 0.00,
                        4: 0.00, 5: 0.00, 6: 0.00
                    }
                ).rename(self.klass._attribute_feature_name('attr1')),
                pd.Series(
                    {
                        0: 0.00, 1: 1.00, 2: 2.00, 3: 3.00,
                        4: 4.00, 5: 5.00, 6: 6.00
                    }
                ).rename(self.klass._attribute_feature_name('attr2')),
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
                ).rename('external_edges'),          
            ]
            expected_features = pd.concat(features, axis=1)

            table = {
                'attributes=True': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                    },
                    'expected_features': expected_features,
                },
                'no attributes kwargs': {
                    'graph': self.graph_attrs,
                    'kwargs': {},
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr1'),
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'attributes=True, graph has no attributes': {
                    'graph': self.graph,
                    'kwargs': {
                        'attributes': True,
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr1'),
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'include list with all attrs': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_include': ['attr1', 'attr2'],
                    },
                    'expected_features': expected_features,
                },
                'include list with some attrs': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_include': ['attr1'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'exclude list with all attrs': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_exclude': ['attr1', 'attr2'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr1'),
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'exclude list with some attrs': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_exclude': ['attr2'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'include and exclude lists': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_include': ['attr1'],
                        'attributes_exclude': ['attr2'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'include and exclude lists conflict': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_include': ['attr1', 'attr2'],
                        'attributes_exclude': ['attr2'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
                'include and exclude lists mutually exclusive': {
                    'graph': self.graph_attrs,
                    'kwargs': {
                        'attributes': True,
                        'attributes_include': ['attr2'],
                        'attributes_exclude': ['attr2'],
                    },
                    'expected_features': expected_features.drop([
                        self.klass._attribute_feature_name('attr1'),
                        self.klass._attribute_feature_name('attr2'),
                    ], axis=1),
                },
            }

            for test_name, test in table.items():
                test_graph = self.klass(test['graph'], **test['kwargs'])
                result_features = test_graph.get_neighborhood_features()
                try:
                    pd.testing.assert_frame_equal(result_features, test['expected_features'])
                except AssertionError as err:
                    raise assertion_error_with_test_case(err, test_name)

class TestNetworkxInterface(BaseGraphInterfaceTest.BaseGraphInterfaceTestCases):

    """ Unit tests for Networkx interface """

    @classmethod
    def setUpClass(cls):
        cls.klass = interface.NetworkxInterface
        
        G_empty = nx.Graph()
        
        G = nx.Graph(cls.edges)

        G_directed_weighted = nx.DiGraph()
        for edge, weight in zip(cls.edges, cls.weights):
            G_directed_weighted.add_edge(*edge, weight=weight)

        G_attrs = G.copy()
        nx.set_node_attributes(G_attrs, cls.attrs)
        
        cls.graph_empty = G_empty
        cls.graph = G
        cls.graph_directed_weighted = G_directed_weighted
        cls.graph_attrs = G_attrs

class TestIgraphInterface(BaseGraphInterfaceTest.BaseGraphInterfaceTestCases):

    """ Unit tests for Igraph interface """

    @unittest.skipIf(
        skip_igraph_tests,
        reason='igraph not found, skipping associated interface tests'
    )
    @classmethod
    def setUpClass(cls):
        cls.klass = interface.IgraphInterface
        
        G_empty = ig.Graph()     

        G = ig.Graph()
        G.add_vertices(len(cls.nodes))
        G.add_edges(cls.edges)
        
        G_directed_weighted = ig.Graph(directed=True)
        G_directed_weighted.add_vertices(len(cls.nodes))
        for edge, weight in zip(cls.edges, cls.weights):
            G_directed_weighted.add_edge(*edge, weight=weight)
        
        G_attrs = G.copy()
        for node_idx, attrs in cls.attrs.items():
            for attr_name, attr_val in attrs.items():
                G_attrs.vs[node_idx][attr_name] = attr_val

        cls.graph_empty = G_empty
        cls.graph = G
        cls.graph_directed_weighted = G_directed_weighted
        cls.graph_attrs = G_attrs
