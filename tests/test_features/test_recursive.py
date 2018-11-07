import unittest

import networkx as nx
import numpy as np
import pandas as pd

from graphrole.features.recursive import RecursiveFeatureExtractor


class TestRecursiveFeatureExtractor(unittest.TestCase):

    """ Unit tests for RecursiveFeatureExtractor """

    edges = [
        ('a', 'b'), ('a', 'c'), ('c', 'd'),
    ]

    G = nx.Graph(edges)
    
    aggs = [
        np.sum,
        np.mean
    ]

    def setUp(self):
        self.rfe = RecursiveFeatureExtractor(self.G, aggs=self.aggs)

    def test__get_next_features_empty_graph(self):
        self.rfe = RecursiveFeatureExtractor(nx.Graph())
        features = self.rfe._get_next_features()
        self.assertTrue(features.empty)
    
    def test__get_next_features(self):
        # generation 0
        features_gen0 = self.rfe._get_next_features()
        expected_features_gen0 = {
            'degree':         {'a': 2, 'b': 1, 'c': 2, 'd': 1},
            'internal_edges': {'a': 2, 'b': 1, 'c': 2, 'd': 1},
            'external_edges': {'a': 1, 'b': 1, 'c': 1, 'd': 1}
        }
        self.assertTrue(features_gen0.equals(pd.DataFrame(expected_features_gen0)))
        
        # generation > 0
        self.rfe.generation_count = 1
        self.rfe.generation_dict[0] = set(features_gen0.columns)
        self.rfe.features = features_gen0
        features_gen1 = self.rfe._get_next_features()
        expected_features_gen1 = {
            'external_edges(sum)':  {'a': 2.0, 'b': 1.0, 'c': 2.0, 'd': 1.0},
            'degree(sum)':          {'a': 3.0, 'b': 2.0, 'c': 3.0, 'd': 2.0},
            'internal_edges(sum)':  {'a': 3.0, 'b': 2.0, 'c': 3.0, 'd': 2.0},
            'external_edges(mean)': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0},
            'degree(mean)':         {'a': 1.5, 'b': 2.0, 'c': 1.5, 'd': 2.0},
            'internal_edges(mean)': {'a': 1.5, 'b': 2.0, 'c': 1.5, 'd': 2.0}
        }
        self.assertTrue(np.allclose(
            features_gen1.sort_index(axis=1),
            pd.DataFrame(expected_features_gen1).sort_index(axis=1)
        ))
    
    def test__update(self):
        self.rfe.feature_group_thresh = -1  # has the effect of disabling pruning
        # seed with existing features
        existing_features = self.rfe._get_next_features()
        self.rfe.final_features = [existing_features]
        self.rfe.final_features_names = set(existing_features.columns)
        # update with new features, include some overlap with old features
        new_features = [
            existing_features[existing_features.columns[0]],
            pd.DataFrame(
                np.random.rand(existing_features.shape[0], 2),
                columns=['a', 'b'],
                index=existing_features.index
            )
        ]
        new_features = pd.concat(new_features, axis=1)
        self.rfe._update(new_features)
        # test final features
        expected_new_final_features = new_features[['a', 'b']]
        expected_new_final_feature_names = \
            set(existing_features.columns).union(set(new_features.columns))
        self.assertTrue(self.rfe.final_features[-1].equals(expected_new_final_features))
        self.assertSetEqual(self.rfe.final_features_names, expected_new_final_feature_names)

    def test__prune_features(self):
        pass

    def test__get_oldest_feature(self):
        pass
    
    def test__add_features(self):
        pass
    
    def test__drop_features(self):
        pass

    def test__aggregated_df_to_dict(self):
        pass

    def test__set_getitem(self):
        table = {
            'ints': {
                'input': {3, 2, 5, 6},
                'expected': 2
            },
            'strings': {
                'input': {'d', 'b', 'a', 'c'},
                'expected': 'a'
            }
        }
        n_trials = 10
        for test_name, test in table.items():
            for _ in range(n_trials):
                result = self.rfe._set_getitem(test['input'])
                self.assertEqual(result, test['expected'], test_name)
