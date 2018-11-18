import itertools as it
from collections import defaultdict
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

VectorLike = Union[np.array, pd.Series]
MatrixLike = Union[np.ndarray, pd.DataFrame]

NodeName = Union[int, str]   # TODO: should be imported from graph.graph?
Edge = Tuple[NodeName, NodeName]


def group_features(binned_features: MatrixLike, dist_thresh: int = 0) -> Iterator[Set[NodeName]]:
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


def _get_connected_components_from_edges(edges: Iterable[Edge]) -> Iterator[Set[NodeName]]:
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


def _get_adj_dict(edges: Iterable[Edge]) -> Dict[NodeName, Set[NodeName]]:
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
    adj_dict: Dict[NodeName, Set[NodeName]],
    node: NodeName,
    visited: Optional[Set[NodeName]] = None
) -> Set[NodeName]:
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
