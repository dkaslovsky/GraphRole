import itertools as it
from typing import Iterator, Set, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from graphrole.graph.graph import AdjacencyDictGraph
from graphrole.graph.interface import NodeName

MatrixLike = Union[np.ndarray, pd.DataFrame]


def group_features(binned_features: MatrixLike, dist_thresh: int = 0) -> Iterator[Set[NodeName]]:
    """
    Group features according to connected components of graph
    induced by pairwise distances below distance threshold
    :param binned_features: np.ndarray of logarithmically binned features
    :param dist_thresh: threshold below which features are to be connected by an edge
    """
    # condensed vector of pairwise distances measuring
    # max_i |u[i] - v[i]| for features u, v
    dists = pdist(binned_features.T, metric='chebychev')

    # construct feature graph by connecting features within
    # dist_thresh of each other and return connected components
    try:
        nodes = binned_features.columns
    except AttributeError:
        n_features = binned_features.shape[1]
        nodes = range(n_features)

    # construct graph of features
    all_edges = it.combinations(nodes, 2)
    edges = it.compress(all_edges, dists <= dist_thresh)
    feature_graph = AdjacencyDictGraph(edges)
    
    # return features grouped as connected components
    groups = feature_graph.get_connected_components()
    return groups
