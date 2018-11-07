import unittest

import networkx as nx
import numpy as np
import pandas as pd

from graphrole.features.similarity import vertical_log_binning
from graphrole.features.recursive import RecursiveFeatureExtractor


# pylint: disable=W0212

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
        features = self.rfe._get_next_features()
        feature_names = features.columns
        
        table = {
            'empty list': {
                'to_drop': [],
                'expected': feature_names
            },
            'one feature dropped': {
                'to_drop': feature_names[0],
                'expected': feature_names[1:]
            },
            'two features dropped': {
                'to_drop': feature_names[:2],
                'expected': feature_names[2:]
            },
            'all features dropped': {
                'to_drop': feature_names,
                'expected': []
            },
        }

        for test_name, test in table.items():
            self.setUp()
            self.rfe._add_features(features)
            self.rfe._drop_features(test['to_drop'])
            self.assertSetEqual(
                set(self.rfe.features.columns),
                set(test['expected']),
                test_name
            )
            self.assertSetEqual(
                set(self.rfe.binned_features.columns),
                set(test['expected']),
                test_name
            )

    def test__aggregated_df_to_dict(self):
        # dataframe
        index = ['sum', 'mean']
        columns = ['feature1', 'feature2', 'feature3']
        data = np.arange(len(index) * len(columns)).reshape(len(index), len(columns))
        df = pd.DataFrame(data, columns=columns, index=index)
        agg_dict = self.rfe._aggregated_df_to_dict(df)
        expected_agg_dict = {
            'feature1(sum)':  0,
            'feature2(sum)':  1,
            'feature3(sum)':  2,
            'feature1(mean)': 3,
            'feature2(mean)': 4,
            'feature3(mean)': 5,
        }
        self.assertDictEqual(agg_dict, expected_agg_dict)
        
        # TODO: THIS TEST FAILS!
        # series
        series = df.iloc[0]
        agg_dict = self.rfe._aggregated_df_to_dict(series)
        expected_agg_dict = {
            'feature1(sum)':  0,
            'feature2(sum)':  1,
            'feature3(sum)':  2,
        }
        self.assertDictEqual(agg_dict, expected_agg_dict)

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
