import unittest

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from graphrole.features.recursive import RecursiveFeatureExtractor

np.random.seed(0)

# pylint: disable=W0212

class BaseRecursiveFeatureExtractorTest:

    class TestCases(unittest.TestCase):

        """ Unit tests for RecursiveFeatureExtractor """

        G = None
        G_empty = None

        nodes = ['a', 'b', 'c', 'd']
        edges = [('a', 'b'), ('a', 'c'), ('c', 'd')]

        aggs = [
            np.sum,
            np.mean
        ]

        def setUp(self):
            self.rfe = RecursiveFeatureExtractor(self.G, aggs=self.aggs)

        def test_initialize_with_unknown_graph_type(self):
            class SomeGraph:
                pass
            with self.assertRaises(TypeError):
                _ = RecursiveFeatureExtractor(SomeGraph)

        def test__get_next_features_empty_graph(self):
            self.rfe = RecursiveFeatureExtractor(self.G_empty)
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
            # some graph interfaces do not support string node names so we will test
            # the values of the feature DataFrames and intentionally ignore the index
            self.assertTrue(np.allclose(
                features_gen0.sort_index(axis=1).sort_index(axis=0).values, 
                pd.DataFrame(expected_features_gen0).sort_index(axis=1).sort_index(axis=0).values
            ))

            # generation > 0
            self.rfe.generation_count = 1
            self.rfe.generation_dict[0] = set(features_gen0.columns)
            self.rfe._features = features_gen0
            features_gen1 = self.rfe._get_next_features()
            expected_features_gen1 = {
                'external_edges(sum)':  {'a': 2.0, 'b': 1.0, 'c': 2.0, 'd': 1.0},
                'degree(sum)':          {'a': 3.0, 'b': 2.0, 'c': 3.0, 'd': 2.0},
                'internal_edges(sum)':  {'a': 3.0, 'b': 2.0, 'c': 3.0, 'd': 2.0},
                'external_edges(mean)': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0},
                'degree(mean)':         {'a': 1.5, 'b': 2.0, 'c': 1.5, 'd': 2.0},
                'internal_edges(mean)': {'a': 1.5, 'b': 2.0, 'c': 1.5, 'd': 2.0}
            }
            # some graph interfaces do not support string node names so we will test
            # the values of the feature DataFrames and intentionally ignore the index
            self.assertTrue(np.allclose(
                features_gen1.sort_index(axis=1).sort_index(axis=0).values,
                pd.DataFrame(expected_features_gen1).sort_index(axis=1).sort_index(axis=0).values
            ))

        def test__update(self):
            self.rfe._feature_group_thresh = -1  # has the effect of disabling pruning
            # seed with existing features
            existing_features = self.rfe._get_next_features()
            self.rfe._final_features = [existing_features]
            self.rfe._final_features_names = set(existing_features.columns)
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
            pd.testing.assert_frame_equal(self.rfe._final_features[-1], expected_new_final_features)
            self.assertSetEqual(self.rfe._final_features_names, expected_new_final_feature_names)

        def test__add_features(self):
            generation_count = 2
            # build feature dataframe
            feature_names = ['a', 'b', 'c']
            nodes = ['node1', 'node2']
            data = np.random.rand(len(nodes), len(feature_names))
            features = pd.DataFrame(data, columns=feature_names, index=nodes)
            # generation dict should only have a key for generation_count
            expected_generation_dict = {generation_count: set(features.columns)}

            # set generation_count
            self.rfe.generation_count = generation_count
            # add features
            self.rfe._add_features(features)
            # test features and generation_dict
            pd.testing.assert_frame_equal(self.rfe._features, features)
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
                    set(self.rfe._features.columns),
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

            # series
            series = pd.Series([6, 7, 8], index=columns, name='prod')
            agg_dict = self.rfe._aggregated_df_to_dict(series)
            expected_agg_dict = {
                'feature1(prod)': 6,
                'feature2(prod)': 7,
                'feature3(prod)': 8,
            }
            self.assertDictEqual(agg_dict, expected_agg_dict)

        def test__finalize_features(self):
            # construct expected result
            data = {
                'node1': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4},
                'node2': {'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9},
            }
            expected_final_features = pd.DataFrame.from_dict(data, orient='index')
            # seed with list of pieces of result
            self.rfe._final_features = [
                expected_final_features[['a', 'b']],
                expected_final_features[['c', 'd']],
                expected_final_features['e'],
            ]
            # test
            final_features = self.rfe._finalize_features()
            pd.testing.assert_frame_equal(final_features, expected_final_features)

        def test_extract_features_back_to_back(self):
            features1 = self.rfe.extract_features()
            features2 = self.rfe.extract_features()
            pd.testing.assert_frame_equal(features1, features2)


class TestRecursiveFeatureExtractorNetworkx(BaseRecursiveFeatureExtractorTest.TestCases):

    """ Unit tests using Networkx interface """

    @classmethod
    def setUpClass(cls):
        cls.G = nx.Graph(cls.edges)
        cls.G_empty = nx.Graph()


class TestRecursiveFeatureExtractorIgraph(BaseRecursiveFeatureExtractorTest.TestCases):

    """ Unit tests using Igraph interface """

    @classmethod
    def setUpClass(cls):
        G = ig.Graph()
        G.add_vertices(len(cls.nodes))
        G.vs()['name'] = cls.nodes
        G.add_edges(cls.edges)
        cls.G = G
        cls.G_empty = ig.Graph()


class TestWithDanglingNode(unittest.TestCase):

    """ Unit tests for RecursiveFeatureExtractor when graph has dangling nodes """

    def setUp(self):
        # build graph with dangling nodes
        self.nodes = ['a', 'b', 'c', 'd']
        self.edge = ('a', 'c')
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edge(*self.edge)
        self.rfe = RecursiveFeatureExtractor(self.G)

    def test_e2e_with_dangling_nodes(self):
        features = self.rfe.extract_features()
        # test that all nodes are present in feature dataframe
        self.assertListEqual(features.index.tolist(), self.nodes)
        # test that no features are null/nan
        self.assertTrue(all(features.notnull()))

    def test_internal_with_dangling_nodes(self):
        # manually simulate one generation
        next_features0 = self.rfe._get_next_features()
        self.rfe._features = next_features0
        self.rfe.generation_dict[self.rfe.generation_count] = next_features0.columns
        self.rfe.generation_count += 1
        # get next features
        next_features1 = self.rfe._get_next_features()
        # test that dangling nodes did not generate features
        self.assertListEqual(next_features1.index.tolist(), list(self.edge))
        # test that features are not null/nan
        self.assertTrue(all(next_features1.notnull()))
