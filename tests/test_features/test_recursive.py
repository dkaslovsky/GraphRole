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
        self.assertTrue(np.allclose(features_gen1, pd.DataFrame(expected_features_gen1)))