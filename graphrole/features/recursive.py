import itertools as it
from typing import Dict, Iterable, Iterator, List, Optional, Set, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from graphrole.features.binning import vertical_log_binning
from graphrole.graph import interface
from graphrole.graph.graph import AdjacencyDictGraph

DataFrameLike = Union[pd.DataFrame, pd.Series]
T = TypeVar('T', int, str)  # generic for feature name


class RecursiveFeatureExtractor:

    """ Compute recursive features for nodes of a graph """

    supported_graph_libs = interface.get_supported_graph_libraries()

    default_aggs = [
        pd.DataFrame.sum,
        pd.DataFrame.mean,
    ]

    def __init__(self,
        G: interface.GraphLibInstance,
        max_generations: int = 10,
        aggs: Optional[List] = None
    ) -> None:
        """
        :param G: graph object from supported graph package
        :param max_generations: maximum levels of recursion
        :param aggs: optional list of aggregations for each recursive generation
        """

        graph = interface.get_interface(G)
        if graph is None:
            raise TypeError(f'Input graph G must be from one of the following '
                            f'supported libraries: {self.supported_graph_libs}')

        self.graph = graph
        self.max_generations = max_generations
        self.aggs = aggs if aggs else self.default_aggs

        # current number of recursive generations
        self.generation_count = 0
        # dict mapping recursive generation number to set of features generated
        self.generation_dict = {}  # type: Dict[int, Set[str]]

        # distance threshold for grouping (binned) features; incremented
        # by one at each generation, so although it always matches
        # self.generation_count, we maintain it as a separate instance
        # variable for clarity
        self._feature_group_thresh = 0
        
        # pd.DataFrame holding current features and their binned counterparts;
        # note that we could recompute binned_features at each generation rather
        # than store them in an instance variable, but this potential memory
        # optimization is not yet needed
        self._features = None            # type: Optional[DataFrameLike]
        self._binned_features = None     # type: Optional[DataFrameLike]

        # list of pd.DataFrames to be concatenated representing the features retained
        # at each generation to be emitted as the final extracted features
        self._final_features = []           # type: List[Union[pd.DataFrame, pd.Series]]
        # feature names of the features stored in the list of DataFrames (self.final_features)
        # used mainly for cheap deduplication of retained features
        self._final_features_names = set()  # type: Set[str]

    def extract_features(self) -> DataFrameLike:
        """
        Perform recursive feature extraction to return DataFrame of features
        """
        for generation in range(self.max_generations):
            
            self.generation_count = generation
            self._feature_group_thresh = generation

            next_features = self._get_next_features()
            self._update(next_features)

            # stop if a recursive iteration results in no features retained
            retained_features = (
                self.generation_dict[generation].intersection(self._features.columns)
            )
            if not retained_features:
                return self._finalize_features()
   
        return self._finalize_features()

    def _finalize_features(self) -> DataFrameLike:
        """
        Return concatenated DataFrame of final features
        """
        return pd.concat(self._final_features, axis=1)

    def _get_next_features(self) -> DataFrameLike:
        """
        Return next level of recursive features (aggregations of node
        features from previous generation)
        """
        # initial (generation 0) features are neighborhood features
        if self.generation_count == 0:
            return self.graph.get_neighborhood_features()

        # get nodes neighbors and aggregate their previous generation features
        prev_features = self.generation_dict[self.generation_count - 1]
        # take intersection with remaining features to avoid aggregated a removed/pruned feature
        prev_features = prev_features.intersection(self._features.columns)
        features = {
            node: (
                self._features
                # select previous generation features for neighbors of current node
                .reindex(index=self.graph.get_neighbors(node), columns=prev_features)
                # aggregate
                .agg(self.aggs)
                # fill nans that result from dangling nodes with 0
                .fillna(0)
                # store new aggregations as dict
                .pipe(self._aggregated_df_to_dict)
            )
            for node in self.graph.get_nodes()
        }
        return pd.DataFrame.from_dict(features, orient='index')

    def _update(self, features: DataFrameLike) -> None:
        """
        Add current generation features and prune across all features to emit final
        features from current generation
        :param features: candidate features from current recursive generation
        """
        self._add_features(features)
        self._prune_features()
        # save features that remain after pruning
        # and that have not previously been saved
        new_features = (
            self._features.columns
            .difference(self._final_features_names)
        )
        self._final_features.append(self._features[new_features])
        self._final_features_names.update(new_features)

    def _prune_features(self) -> None:
        """
        Eliminate redundant features from current iteration by identifying
        features in connected components of a feature graph and replace components
        with oldest (i.e., earliest generation) member feature
        """
        features_to_drop = []
        groups = self._group_features()
        for group in groups:
            oldest = self._get_oldest_feature(group)
            to_drop = group - {oldest}
            features_to_drop.extend(to_drop)
        self._drop_features(features_to_drop)
    
    def _group_features(self) -> Iterator[Set[interface.Node]]:
        """
        Group features according to connected components of feature graph induced
        by pairwise distances below distance threshold
        """
        # get condensed vector of pairwise distances measuring
        # max_i |u[i] - v[i]| for features u, v
        dists = pdist(self._binned_features.T, metric='chebychev')
        # construct feature graph by connecting features within
        # dist_thresh of each other and return connected components
        nodes = self._binned_features.columns
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
        for gen in range(self.generation_count):
            cur_gen = feature_names.intersection(self.generation_dict[gen])
            if cur_gen:
                return self._set_getitem(cur_gen)
        return self._set_getitem(feature_names)

    def _add_features(self, features: DataFrameLike) -> None:
        """
        Add features to self.features DataFrame; also update corresponding
        binned_features DataFrame and feature generation_dict
        :param features: DataFrame of features to be added
        """
        binned_features = features.apply(vertical_log_binning)
        self._features = pd.concat(
            [self._features, features], axis=1, sort=True
        )
        self._binned_features = pd.concat(
            [self._binned_features, binned_features], axis=1, sort=True
        )
        self.generation_dict[self.generation_count] = set(features.columns)
    
    def _drop_features(self, feature_names: Iterable[str]) -> None:
        """
        Drop feature_names from feature and binned feature DataFrames
        :param feature_names: iterable of feature names to be dropped
        """
        self._features.drop(feature_names, axis=1, inplace=True)
        self._binned_features.drop(feature_names, axis=1, inplace=True)

    @staticmethod
    def _aggregated_df_to_dict(agg_df: DataFrameLike) -> Dict[str, float]:
        """
        Transform DataFrame of aggregated features to dict formatted for
        concatenation with self.features DataFrame
        :param agg_df: agregated features resulting from df.agg(self.aggs)
        """
        try:
            agg_dicts = agg_df.to_dict(orient='index')
        except TypeError:
            agg_dicts = agg_df.to_frame().T.to_dict(orient='index')
        return {
            f'{key}({idx})': val
            for idx, row in agg_dicts.items()
            for key, val in row.items()
        }

    @staticmethod
    def _set_getitem(s: Set[T]) -> T:
        """
        Cast set to list and return first element after sorting to ensure
        deterministic, repeatable getitem functionality from set
        :param s: set
        """
        return np.partition(list(s), 0)[0]
