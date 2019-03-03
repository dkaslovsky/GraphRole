import unittest

import numpy as np
import pandas as pd

from graphrole.features.prune import FeaturePruner, vertical_log_binning

np.random.seed(0)


# pylint: disable=protected-access

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
            # test numpy array
            numpy_array_result = vertical_log_binning(test['input'], frac=frac)
            numpy_msg = f'{test_name} numpy array'
            np.testing.assert_array_equal(
                numpy_array_result,
                test['expected'],
                err_msg=numpy_msg
            )
            # test pandas series
            pandas_series_result = vertical_log_binning(pd.Series(test['input']), frac=frac)
            pandas_msg = f'{test_name} pandas series'
            np.testing.assert_array_almost_equal(
                pandas_series_result,
                test['expected'],
                err_msg=pandas_msg
            )


class TestFeaturePruner(unittest.TestCase):

    """ Unit test for FeaturePruner """

    def setUp(self):
        generation_dict = {
            0: {'b': {0: 0, 1: 1}, 'a': {0: 2, 1: 3}},
            1: {'c': {0: 4, 1: 5}, 'd': {0: 6, 1: 7}}
        }
        feature_group_thresh = 1
        self.pruner = FeaturePruner(generation_dict, feature_group_thresh)

    def test_prune_features(self):
        data = {
            'a': [1, 2, 3, 10],
            'b': [1, 2, 3, 1],
            'c': [2, 1, 1, 4],
            'd': [1, 1, 1, 1],
            'e': [1, 1, 2, 0]
        }
        features = pd.DataFrame(data)

        generation_dict = {
            0: {'a': {0: 0, 1: 1}, 'b': {0: 2, 1: 3}, 'c': {0: 8, 1: 9}},
            1: {'d': {0: 4, 1: 5}, 'e': {0: 6, 1: 7}}
        }
        self.pruner._generation_dict = generation_dict

        table = {
            'no pruning': {
                'feature_group_thresh': 0,
                'expected_features_to_drop': []
            },
            'two groups': {
                'feature_group_thresh': 1,
                'expected_features_to_drop': ['c', 'd', 'e']
            },
            'one group': {
                'feature_group_thresh': 2,
                'expected_features_to_drop': ['b', 'c', 'd', 'e']
            },
        }
        for test_name, test in table.items():
            self.pruner._feature_group_thresh = test['feature_group_thresh']
            expected = test['expected_features_to_drop']
            features_to_drop = self.pruner.prune_features(features)
            self.assertSetEqual(set(features_to_drop), set(expected), test_name)

    def test__group_features(self):
        data = {
            'a': [1, 2, 3],
            'b': [1, 2, 3],
            'c': [2, 1, 1],
            'd': [1, 1, 1]
        }
        features = pd.DataFrame(data)

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
        for test_name, test in table.items():
            self.pruner._feature_group_thresh = test['dist_thresh']
            groups = self.pruner._group_features(features)
            self.assertListEqual(list(groups), test['expected'], test_name)

    def test__get_oldest_feature(self):
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
                'feature_names': {'x', 'd', 'f', 'aa'},
                'expected': 'd'
            },
            'no gen0 or gen1 features as input': {
                'feature_names': {'y', 'x', 'z'},
                'expected': 'x'
            }
        }
        for test_name, test in table.items():
            oldest = self.pruner._get_oldest_feature(test['feature_names'])
            self.assertEqual(oldest, test['expected'], test_name)

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
                result = self.pruner._set_getitem(test['input'])
                self.assertEqual(result, test['expected'], test_name)
