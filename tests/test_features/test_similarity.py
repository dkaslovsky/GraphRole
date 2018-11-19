import unittest

import numpy as np
import pandas as pd

from graphrole.features.similarity import group_features


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
            groups = group_features(self.binned_features, dist_thresh=dist_thresh)
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
            groups = group_features(binned_features_df, dist_thresh=dist_thresh)
            self.assertEqual(list(groups), test['expected'], test_name)
