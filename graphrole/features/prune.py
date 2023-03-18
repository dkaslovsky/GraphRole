import itertools as it
from typing import Dict, Iterator, List, Set, TypeVar

import numpy as np
from scipy.spatial.distance import pdist

from graphrole.graph.graph import AdjacencyDictGraph
from graphrole.types import DataFrameDict, DataFrameLike, VectorLike

T = TypeVar('T', int, str)


def vertical_log_binning(arr: VectorLike, frac: float = 0.5) -> VectorLike:
    """
    Reassigns values of an array into vertical logarithmic bins
    :param arr: array to be binned
    :param frac: value in (0, 1) defining fraction of values assigned to each bin
    """

    if not 0 < frac < 1:
        raise ValueError('must specify frac in interval (0, 1)')

    arr_len = len(arr)
    binned = np.zeros(arr_len, dtype=int)

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


class FeaturePruner:

    """ Determines redundant features to be removed from future recursive aggregations """

    def __init__(
        self,
        generation_dict: Dict[int, DataFrameDict],
        feature_group_thresh: int
    ) -> None:
        """
        :param generation_dict: mapping of recursive generation number to
        dict of {features: {node: values}}
        :param feature_group_thresh: distance threshold for grouping binned version of features
        """
        self._generation_dict = generation_dict
        self._feature_group_thresh = feature_group_thresh

    def prune_features(self, features: DataFrameLike) -> List[str]:
        """
        Eliminate redundant features from current iteration by identifying
        features in connected components of a feature graph and replace components
        with oldest (i.e., earliest generation) member feature
        :param features: DataFrame of features
        """
        features_to_drop = []
        groups = self._group_features(features)
        for group in groups:
            # isolated feature should not be pruned
            if len(group) == 1:
                continue
            oldest = self._get_oldest_feature(group)
            to_drop = group - {oldest}
            features_to_drop.extend(to_drop)
        return features_to_drop

    def _group_features(self, features: DataFrameLike) -> Iterator[Set[str]]:
        """
        Group features according to connected components of feature graph induced
        by pairwise distances below distance threshold
        :param features: DataFrame of features
        """
        # apply binning to features
        # note that some (non-pruned) features will be rebinned each time when this class is
        # used for pruning multiple generations of features, but this slight inefficiency removes
        # maintaining binned features in the state of the feature extraction class and is thus an
        # intentional tradeoff
        binned_features = features.apply(vertical_log_binning)
        # get condensed vector of pairwise distances measuring
        # max_i |u[i] - v[i]| for features u, v
        dists = pdist(binned_features.T, metric='chebychev')
        # construct feature graph by connecting features within
        # dist_thresh of each other and return connected components
        nodes = binned_features.columns
        all_edges = it.combinations(nodes, 2)
        edges = it.compress(all_edges, dists <= self._feature_group_thresh)
        feature_graph = AdjacencyDictGraph(edges)
        groups = feature_graph.get_connected_components()
        return groups

    def _get_oldest_feature(self, feature_names: Set[T]) -> T:
        """
        Return the feature from set of feature names that was generated
        in the earliest generation; tie between features from same iteration
        are broken by sorted named order
        :param feature_names: set of feature names from which to find oldest
        """
        for gen in range(len(self._generation_dict)):
            generation_features = self._generation_dict[gen].keys()
            cur_features = feature_names.intersection(generation_features)
            if cur_features:
                return self._set_getitem(cur_features)
        return self._set_getitem(feature_names)

    @staticmethod
    def _set_getitem(s: Set[T]) -> T:
        """
        Cast set to list and return first element after sorting to ensure
        deterministic, repeatable getitem functionality from set
        :param s: set
        """
        return np.partition(list(s), 0)[0]
