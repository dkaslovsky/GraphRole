import unittest

import numpy as np
import pandas as pd

from graphrole.features.binning import vertical_log_binning


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
            self.assertTrue(np.allclose(numpy_array_result, test['expected']), numpy_msg)
            # test pandas series
            pandas_series_result = vertical_log_binning(pd.Series(test['input']), frac=frac)
            pandas_msg = f'{test_name} pandas series'
            self.assertTrue(np.allclose(pandas_series_result, test['expected']), pandas_msg)
