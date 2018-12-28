from typing import Dict, Iterable, List, Optional, TypeVar

import pandas as pd

from graphrole.features.prune import DataFrameLike, FeaturePruner
from graphrole.graph import interface

T = TypeVar('T', int, str)


class RecursiveFeatureExtractor:

    """ Compute recursive features for nodes of a graph """

    supported_graph_libs = interface.get_supported_graph_libraries()

    default_aggs = [
        pd.DataFrame.sum,
        pd.DataFrame.mean,
    ]

    def __init__(
        self,
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
        
        # pd.DataFrame holding current
        self._features = None     # type: Optional[DataFrameLike]

        # dict of feature names to feature values representing the features retained
        # at each generation to be emitted as the final extracted features
        self._final_features = {}           # type: XXXX

    def extract_features(self) -> DataFrameLike:
        """
        Perform recursive feature extraction to return DataFrame of features
        """
        # return already calculated features if stored in state
        if self._final_features:
            return self._finalize_features()

        for generation in range(self.max_generations):

            self.generation_count = generation
            self._feature_group_thresh = generation

            features = self._get_next_features()
            self._update(features)

            # stop if a recursive iteration results in no features retained
            cur_features = self.generation_dict[generation]
            retained_features = cur_features.intersection(self._features.columns)
            if not retained_features:
                return self._finalize_features()

        return self._finalize_features()

    def _finalize_features(self) -> DataFrameLike:
        """
        Return concatenated DataFrame of final features
        """
        #return pd.concat(self._final_features, axis=1)
        return pd.DataFrame(self._final_features)

    def _get_next_features(self) -> DataFrameLike:
        """
        Return next level of recursive features (aggregations of node
        features from previous generation)
        """
        # initial (generation 0) features are neighborhood features
        if self.generation_count == 0:
            return self.graph.get_neighborhood_features()

        # get nodes neighbors and aggregate their previous generation features, taking
        # intersection with remaining features to avoid aggregated a removed/pruned feature
        prev_features = self.generation_dict[self.generation_count - 1]
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

        # prune redundant features
        pruner = FeaturePruner(self.generation_dict, self._feature_group_thresh)
        features_to_prune = pruner.prune_features(self._features)
        self._drop_features(features_to_prune)

        # save features that remain after pruning and that
        # have not previously been saved as final features
        retained = features.columns.difference(features_to_prune)
        self._final_features.update(self._features[retained].to_dict())

    def _add_features(self, features: DataFrameLike) -> None:
        """
        Add features to self.features DataFrame; also update generation_dict
        :param features: DataFrame of features to be added
        """
        self._features = pd.concat([self._features, features], axis=1, sort=True)
        self.generation_dict[self.generation_count] = set(features.columns)

    def _drop_features(self, feature_names: Iterable[str]) -> None:
        """
        Drop feature_names from feature and binned feature DataFrames
        :param feature_names: iterable of feature names to be dropped
        """
        self._features.drop(feature_names, axis=1, inplace=True)

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
