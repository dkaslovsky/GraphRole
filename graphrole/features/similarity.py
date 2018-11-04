import itertools as it
from collections import defaultdict
from typing import (Any, Dict, Iterable, Iterator, NewType, Optional, Set,
                    Tuple, Union)

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

Node = NewType('Node', Union[int, str])


def vertical_log_binning(
    arr: np.array,
    frac: float = 0.5
) -> np.array:
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
) -> Iterator[Set[Node]]:
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
    try:
        nodes = binned_features.columns
    except AttributeError:
        nodes = range(n_features)
    all_edges = it.combinations(nodes, 2)
    edges = it.compress(all_edges, dists <= dist_thresh)
    groups = _get_connected_components_from_edges(edges)
    return groups


def _get_connected_components_from_edges(
    edges: Iterable[Tuple[Node]]
) -> Iterator[Set[Node]]:
    """
    Generate connected components represented as sets of nodes
    :param edges: iterable of edges represented as (node1, node2) tuples
    """
    # get dict mapping node to neighors
    adj_dict = _get_adj_dict(edges)
    # maintain set of all nodes already accounted for
    visited = set()
    # run dfs over all nodes not previously visited
    for node in adj_dict.keys():
        if node not in visited:
            component = _dfs(adj_dict, node)
            visited.update(component)
            yield component


def _get_adj_dict(
    edges: Iterable[Tuple[Node]]
) -> Dict[Node, Set[Node]]:
    """
    Construct dict mapping node to a set of its neighbor nodes
    :param edges: iterable of edges represented as (node1, node2) tuples
    """
    adj = defaultdict(set)
    for (node1, node2) in edges:
        adj[node1].add(node2)
        adj[node2].add(node1)
    return dict(adj)


def _dfs(
    adj_dict: Dict[Any, Set],
    node: Node,
    visited: Optional[Set] = None
) -> Set:
    """
    Run recursive depth first search starting from node and
    return set of all visited nodes
    :param adj_dict: dict mapping node to set of neighbor nodes
    :param node: node at which to start search
    :param visited: set of all nodes visited, initially should be None
    """
    if not visited:
        visited = set()
    visited.add(node)
    next_level_unvisited = adj_dict[node] - visited
    for nbr in next_level_unvisited:
        _dfs(adj_dict, nbr, visited)
    return visited
