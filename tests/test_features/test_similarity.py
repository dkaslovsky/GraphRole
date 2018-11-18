import unittest

import numpy as np
import pandas as pd

import graphrole.features.similarity as sim


# pylint: disable=W0212

class TestGroupFeatures(unittest.TestCase):

    """ Unit tests for group_features """

    features = [
        np.array([[1, 2, 3]]).T,
        np.array([[1, 2, 3]]).T,
        np.array([[2, 1, 1]]).T,
        np.array([[1, 1, 1]]).T
    ]
    binned_features = np.concatenate(features, axis=1)

    def test_group_features_numpy(self):
        table = {
            'dist_thresh = 0 -> 1 component': {
                'dist_thresh': 0,
                'expected': [{0, 1}]
            },
            'dist_thresh = 1 -> 2 components': {
                'dist_thresh': 1,
                'expected': [{0, 1}, {2, 3}]
            },
            'dist_thresh = 2 -> all connected': {
                'dist_thresh': 2,
                'expected': [{0, 1, 2, 3}]
            },
            'dist_thresh = -1 -> empty list': {
                'dist_thresh': -1,
                'expected': []
            },
        }

        for test_name, test in table.items():
            dist_thresh = test['dist_thresh']
            groups = sim.group_features(self.binned_features, dist_thresh=dist_thresh)
            self.assertEqual(list(groups), test['expected'], test_name)

    def test_group_features_pandas(self):
        table = {
            'dist_thresh = 0 -> 1 component': {
                'dist_thresh': 0,
                'expected': [{'a', 'b'}]
            },
            'dist_thresh = 1 -> 2 components': {
                'dist_thresh': 1,
                'expected': [{'a', 'b'}, {'c', 'd'}]
            },
            'dist_thresh = 2 -> all connected': {
                'dist_thresh': 2,
                'expected': [{'a', 'b', 'c', 'd'}]
            },
            'dist_thresh = -1 -> empty list': {
                'dist_thresh': -1,
                'expected': []
            },
        }

        features = ['a', 'b', 'c', 'd']
        binned_features_df = pd.DataFrame(self.binned_features, columns=features)
        for test_name, test in table.items():
            dist_thresh = test['dist_thresh']
            groups = sim.group_features(binned_features_df, dist_thresh=dist_thresh)
            self.assertEqual(list(groups), test['expected'], test_name)


class TestEdgeBasedConnectedComponents(unittest.TestCase):

    """ Unit tests for _get_adj_dict, _dfs(), and _get_connected_components_from_edges """

    disjoint_nodes = [
        (0, 1), (2, 3)
    ]

    disjoint_nodes_adj = {
        0: {1}, 1: {0}, 2: {3}, 3: {2}
    }

    disjoint_nodes_components = [
        {0, 1}, {2, 3}
    ]

    cycle = [
        (0, 1), (1, 2), (2, 0)
    ]

    cycle_adj = {
        0: {1, 2}, 1: {0, 2}, 2: {0, 1}
    }

    cycle_components = [
        {0, 1, 2}
    ]

    many_edges = [
        (0, 7), (0, 8), (8, 2), (8, 5),
        (1, 3), (6, 2), (6, 4)
    ]

    many_edges_adj = {
        0: {7, 8}, 1: {3}, 2: {6, 8}, 3: {1}, 4: {6},
        5: {8}, 6: {2, 4}, 7: {0}, 8: {0, 2, 5}
    }

    many_edges_components = [
        {0, 2, 4, 5, 6, 7, 8}, {1, 3}
    ]

    def test__get_adj_dict(self):
        table = {
            'disjoint nodes': {
                'edges': self.disjoint_nodes,
                'expected': self.disjoint_nodes_adj
            },
            'cycle': {
                'edges': self.cycle,
                'expected': self.cycle_adj
            },
            'many edges': {
                'edges': self.many_edges,
                'expected': self.many_edges_adj
            },
        }
        for test_name, test in table.items():
            adj_dict = sim._get_adj_dict(test['edges'])
            self.assertDictEqual(adj_dict, test['expected'], test_name)

    def test__dfs(self):
        table = {
            'disjoint nodes': {
                'adj': self.disjoint_nodes_adj,
                'expected components': self.disjoint_nodes_components
            },
            'cycle': {
                'adj': self.cycle_adj,
                'expected components': self.cycle_components
            },
            'many edges': {
                'adj': self.many_edges_adj,
                'expected components': self.many_edges_components
            },
        }
        for test_name, test in table.items():
            adj = test['adj']
            for component in test['expected components']:
                for node in component:
                    result_component = sim._dfs(adj, node)
                    self.assertSetEqual(result_component, component, test_name)

    def test__get_connected_components_from_edges(self):
        table = {
            'disjoint nodes': {
                'edges': self.disjoint_nodes,
                'expected': self.disjoint_nodes_components
            },
            'cycle': {
                'edges': self.cycle,
                'expected': self.cycle_components
            },
            'many edges': {
                'edges': self.many_edges,
                'expected': self.many_edges_components
            },
        }
        for test_name, test in table.items():
            components = sim._get_connected_components_from_edges(test['edges'])
            expected = test['expected']
            for component in components:
                self.assertIn(component, expected, test_name)
