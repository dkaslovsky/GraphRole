import unittest

import numpy as np
import pandas as pd

import graphrole.features.similarity as sim


class TestGroupFeatures(unittest.TestCase):

    """ Unit tests for group_features """

    features = [
        np.array([[1,2,3]]).T,
        np.array([[1,2,3]]).T,
        np.array([[2,1,1]]).T,
        np.array([[1,1,1]]).T
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


class TestVerticalLogBinning(unittest.TestCase):

    """ Unit tests for vertical_log_binning() """
    
    def test_vertical_log_binning(self):
        table = {
            'empty': {
                'input': np.array([]),
                'expected': np.array([])
            },
            'single 0': {
                'input': np.array([0]),
                'expected': np.array([0])
            },
            'single nonzero': {
                'input': np.array([1]),
                'expected': np.array([0])
            },
            'repeated': {
                'input': np.array([1, 1]),
                'expected': np.array([0, 0])
            },
            '2 bins': {
                'input': np.array([1, 2]),
                'expected': np.array([0, 1])
            },
            '2 bins with repeated lower bin': {
                'input': np.array([1, 2, 1]),
                'expected': np.array([0, 1, 0])
            },
            '2 bins with repeated upper bin': {
                'input': np.array([1, 2, 2]),
                'expected': np.array([0, 1, 1])
            },
            'negative and zeros': {
                'input': np.array([-1, 0, 0]),
                'expected': np.array([0, 1, 1])
            },
            '1 through 4': {
                'input': np.array([1, 2, 3, 4]),
                'expected': np.array([0, 0, 1, 2])
            },
            '1 through 5': {
                'input': np.array([1, 2, 3, 4, 5]),
                'expected': np.array([0, 0, 1, 2, 3])
            },
            '1 through 6': {
                'input': np.array([1, 2, 3, 4, 5, 6]),
                'expected': np.array([0, 0, 0, 1, 2, 3])
            },
            'range(10)': {
                'input': np.array(range(10)),
                'expected': np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])
            },
            '-range(10)': {
                'input': -1 * np.array(range(10)),
                'expected': np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4][::-1])
            },
            'non-integer': {
                'input': -0.1 * np.array(range(10)),
                'expected': np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4][::-1])
            },
            'frac=0.1': {
                'input': np.array(range(10)),
                'frac': 0.1,
                'expected': np.array(range(10))
            },
            'frac=0.25': {
                'input': np.array(range(10)),
                'frac': 0.25,
                'expected': np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
            },
        }

        for test_name, test in table.items():
            frac = test.get('frac', 0.5)
            result = sim.vertical_log_binning(test['input'], frac=frac)
            np.testing.assert_array_equal(result, test['expected'], test_name)
