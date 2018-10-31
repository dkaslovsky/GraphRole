import itertools as it
from typing import Iterable, Iterator, Optional, Set, Union

import networkx as nx
import numpy as np
import pandas as pd
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


def group_features(
    binned_features: Union[np.ndarray, pd.DataFrame],
    dist_thresh: int = 0
) -> Iterator[Set[int]]:
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
    if isinstance(binned_features, pd.DataFrame):
        nodes = binned_features.columns
    else:
        nodes = range(n_features)

    all_edges = it.combinations(nodes, 2)
    edges = it.compress(all_edges, dists <= dist_thresh)
    groups = nx.connected_components(nx.Graph(edges))
    return groups
