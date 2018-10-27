import itertools as it
from typing import Iterator, Set

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist


def vertical_log_binning(arr: np.array, frac: float = 0.5) -> np.array:
    """
    Reassigns values of an array into vertical logarithmic bins
    :param arr: array to be binned
    :param frac: value in (0, 1) defining fraction of values assigned to each bin
    """

    if not 0 < frac < 1:
        raise ValueError('must specify frac in interval (0, 1)')

    arr_len = len(arr)
    binned = np.zeros(arr_len, dtype=np.int)
    
    # get sorted unique values and counts in arr
    arr_uniq, counts = np.unique(arr, return_counts=True)
    # convert to cumulative counts
    counts = np.cumsum(counts)

    # initial iteration parameters
    binned_len = 0                  # length of binned portion of arr
    unbinned_len = arr_len          # length of unbinned portion of arr
    bin_min = -np.inf               # left side value of current bin (exclusive)

    for bin_val in range(arr_len):
        
        # bin size is fraction frac of the unbinned len (enforce at least 1)
        bin_size = max(int(frac * unbinned_len), 1)
        # get index of largest unique value from arr to be included in bin
        u_idx = np.searchsorted(counts, binned_len + bin_size)
        bin_max = arr_uniq[u_idx]
        # mark members of current bin with bin_val
        arr_idx = np.logical_and(arr > bin_min, arr <= bin_max)
        binned[arr_idx] = bin_val

        # update iteration paramters
        binned_len += sum(arr_idx)
        unbinned_len = arr_len - binned_len
        bin_min = bin_max
        
        # check if all values have been binned
        if unbinned_len == 0:
            break

    return binned


def group_features(binned_features: np.ndarray, dist_thresh: int = 0) -> Iterator[Set[int]]:
    """
    Group features according to connected components of graph
    induced by pairwise distances below distance threshold
    :param binned_features: np.ndarray of logarithmically binned features
    :param dist_thresh: threshold below which features are to be connected by an edge
    """
    n_features = binned_features.shape[1]

    # condensed vector of pairwise distances measuring
    # max_i |u[i] - v[i]| for features u, v
    dists = pdist(binned_features.T, metric='chebychev')

    # construct feature graph by connecting features within
    # dist_thresh of each other and return connected components
    all_edges = it.combinations(range(n_features), 2)
    edges = it.compress(all_edges, dists <= dist_thresh)
    groups = nx.connected_components(nx.Graph(edges))
    return groups


def test_group_features():
    """
    Test group_features
    """
    
    features = [
        np.array([[1,2,3]]).T,
        np.array([[1,2,3]]).T,
        np.array([[2,1,1]]).T,
        np.array([[1,1,1]]).T
    ]
    binned_features = np.concatenate(features, axis=1)

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
        group = group_features(binned_features, dist_thresh=dist_thresh)
        result = list(group)
        assert result == test['expected'], test_name


def test_vertical_log_binning():
    """
    Test vertical_log_binning()
    """
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
        result = vertical_log_binning(test['input'], frac=frac)
        assert np.all(result == test['expected']), test_name


if __name__ == '__main__':
    test_vertical_log_binning()
    test_group_features()
