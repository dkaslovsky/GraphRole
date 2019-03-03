import unittest

import networkx as nx
import numpy as np
import pandas as pd

from graphrole.features.extract import RecursiveFeatureExtractor, as_frame

# try to import igraph; if not found set flag to skip associated tests
skip_igraph_tests = False
try:
    import igraph as ig
except ImportError:
    skip_igraph_tests = True

np.random.seed(0)


# pylint: disable=protected-access

class TestAsFrame(unittest.TestCase):

    def test_as_frame(self):
        # test series
        series = pd.Series(np.random.rand(10))
        result = as_frame(series)
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd.DataFrame(series))
        # test dataframe
        frame = pd.DataFrame(np.random.rand(10))
        result = as_frame(frame)
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, frame)


class TestRecursiveFeatureExtractorWithDanglingNode(unittest.TestCase):

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
        # manually seed first generation
        next_features0 = self.rfe.graph.get_neighborhood_features()
        self.rfe._features = next_features0
        self.rfe._final_features = {0: next_features0.to_dict()}
        # get next features
        self.rfe.generation_count = 1
        next_features1 = self.rfe._get_next_features()
        # test that dangling nodes did not generate features
        self.assertListEqual(next_features1.index.tolist(), list(self.edge))
        # test that features are not null/nan
        self.assertTrue(all(next_features1.notnull()))


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
            # initialize with neighborhood features
            self.rfe._features = self.rfe.graph.get_neighborhood_features()
            self.rfe._final_features = {0: self.rfe._features.to_dict()}
            self.rfe.generation_count = 1

        def test_initialize_with_unknown_graph_type(self):
            class SomeGraph:
                pass
            with self.assertRaises(TypeError):
                _ = RecursiveFeatureExtractor(SomeGraph)

        def test__initialize_with_empty_graph(self):
            with self.assertRaises(ValueError):
                _ = RecursiveFeatureExtractor(self.G_empty)

        def test__get_next_features(self):
            # self.rfe._features = self.rfe.graph.get_neighborhood_features()
            # self.rfe._final_features = {0: self.rfe._features.to_dict()}
            # self.rfe.generation_count = 1
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
            # update with new features, include one that will be pruned
            existing_features = self.rfe._features
            new_features = pd.concat([
                pd.DataFrame(
                    existing_features['degree'].values,
                    columns=['degree2'],
                    index=existing_features.index
                ),
                pd.DataFrame(
                    np.random.randn(existing_features.shape[0], 2),
                    columns=['a', 'b'],
                    index=existing_features.index
                )
            ], axis=1)

            self.rfe._update(new_features)

            # test _features
            features = self.rfe._features
            expected_features = pd.concat([
                existing_features[['degree', 'external_edges']],
                new_features[['a', 'b']]
            ], axis=1)
            pd.testing.assert_frame_equal(features, expected_features)

            # test _final_features
            final_features = self.rfe._finalize_features()
            expected_final_features = pd.concat([
                existing_features,
                new_features[['a', 'b']]
            ], axis=1)
            pd.testing.assert_frame_equal(
                final_features.sort_index(axis=1),
                expected_final_features.sort_index(axis=1)
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
            self.rfe._final_features = {
                0: expected_final_features[['a', 'b']].to_dict(),
                1: expected_final_features[['c', 'd']].to_dict(),
                2: expected_final_features['e'].to_frame().to_dict(),
            }

            # test
            final_features = self.rfe._finalize_features()
            pd.testing.assert_frame_equal(
                final_features.sort_index(axis=1),
                expected_final_features.sort_index(axis=1)
            )

        def test_extract_features_back_to_back(self):
            self.rfe = RecursiveFeatureExtractor(self.G)
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

    @unittest.skipIf(
        skip_igraph_tests,
        reason='igraph not found, skipping associated extract tests'
    )
    @classmethod
    def setUpClass(cls):
        G = ig.Graph()
        G.add_vertices(len(cls.nodes))
        G.vs()['name'] = cls.nodes
        G.add_edges(cls.edges)
        cls.G = G
        cls.G_empty = ig.Graph()
