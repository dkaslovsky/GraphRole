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
        # seed with 2 generations of features
        for _ in range(2):
            features = self.rfe._get_next_features()
            self.rfe._add_features(features)
            self.rfe.generation_count += 1
        # prune features
        self.rfe._prune_features()
        # test remaining features
        expected_remaining_features = {'degree', 'external_edges', 'degree(mean)'}
        self.assertSetEqual(set(self.rfe.features.columns), expected_remaining_features)
        self.assertSetEqual(set(self.rfe.binned_features.columns), expected_remaining_features)

    def test__get_oldest_feature(self):
        self.rfe.generation_count = 2
        self.rfe.generation_dict = {
            0: {'b', 'a'},
            1: {'c', 'd'}
        }
        table = {
            'gen0': {
                'feature_names': {'a', 'c', 'f'},
                'expected': 'a'
            },
            'gen0 with tie': {
                'feature_names': {'a', 'b', 'f'},
                'expected': 'a'
            },
            'gen1 with features not in generation_dict': {
                'feature_names': {'x', 'c', 'f'},
                'expected': 'c'
            },
            'no gen1 or gen2 features as input': {
                'feature_names': {'y', 'x', 'z'},
                'expected': 'x'
            }
        }
        for test_name, test in table.items():
            oldest = self.rfe._get_oldest_feature(test['feature_names'])
            self.assertEqual(oldest, test['expected'], test_name)

    def test__add_features(self):
        generation_count = 2
        # build feature dataframe
        feature_names = ['a', 'b', 'c']
        nodes = ['node1', 'node2']
        data = np.random.rand(len(nodes), len(feature_names))
        features = pd.DataFrame(data, columns=feature_names, index=nodes)
        # get associated binned_features
        binned_features = features.apply(vertical_log_binning)
        # generation dict should only have a key for generation_count
        expected_generation_dict = {generation_count: set(features.columns)}

        # set generation_count
        self.rfe.generation_count = generation_count
        # add features
        self.rfe._add_features(features)
        # test features, binned_features, and generation_dict
        self.assertTrue(self.rfe.features.equals(features))
        self.assertTrue(self.rfe.binned_features.equals(binned_features))
        self.assertDictEqual(self.rfe.generation_dict, expected_generation_dict)

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

    # def test__aggregated_df_to_dict(self):
    #     # dataframe
    #     index = ['sum', 'mean']
    #     columns = ['feature1', 'feature2', 'feature3']
    #     data = np.arange(len(index) * len(columns)).reshape(len(index), len(columns))
    #     df = pd.DataFrame(data, columns=columns, index=index)
    #     agg_dict = self.rfe._aggregated_df_to_dict(df)
    #     expected_agg_dict = {
    #         'feature1(sum)':  0,
    #         'feature2(sum)':  1,
    #         'feature3(sum)':  2,
    #         'feature1(mean)': 3,
    #         'feature2(mean)': 4,
    #         'feature3(mean)': 5,
    #     }
    #     self.assertDictEqual(agg_dict, expected_agg_dict)
        
    #     # TODO: THIS TEST FAILS!
    #     # series
    #     series = df.iloc[0]
    #     agg_dict = self.rfe._aggregated_df_to_dict(series)
    #     expected_agg_dict = {
    #         'feature1(sum)':  0,
    #         'feature2(sum)':  1,
    #         'feature3(sum)':  2,
    #     }
    #     self.assertDictEqual(agg_dict, expected_agg_dict)

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

    def test__finalize_features(self):
        # construct expected result
        data = {
            'node1': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
            'node2': {'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9},
        }
        expected_final_features = pd.DataFrame.from_dict(data, orient='index')
        # seed with list of pieces of result
        self.rfe.final_features = [
            expected_final_features[['a', 'b']],
            expected_final_features[['c', 'd']],
            expected_final_features['e'],
        ]
        # test
        final_features = self.rfe._finalize_features()
        self.assertTrue(final_features.equals(expected_final_features))
