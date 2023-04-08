from collections import ChainMap
from typing import Dict, List, Optional

import pandas as pd

from graphrole.features.prune import FeaturePruner
from graphrole.graph import interface
from graphrole.types import DataFrameDict, DataFrameLike


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
        aggs: Optional[List] = None,
        **kwargs
    ) -> None:
        """
        :param G: graph object from supported graph package
        :param max_generations: maximum levels of recursion
        :param aggs: optional list of aggregations for each recursive generation
        :kwargs: kwargs accepted by the relevant graph interface
        """

        graph_class = interface.get_interface(G)
        if graph_class is None:
            raise TypeError(f'Input graph G must be from one of the following '
                            f'supported libraries: {self.supported_graph_libs}')
        
        graph = graph_class(G, **kwargs)
        if graph.get_num_edges() == 0:
            raise ValueError('Input graph G must contain at least one edge')

        self.graph = graph
        self.max_generations = max_generations
        self.aggs = aggs if aggs else self.default_aggs

        # current generation
        self.generation_count = 0

        # distance threshold for grouping (binned) features; incremented
        # by one at each generation, so although it always matches
        # self.generation_count, we maintain it as a separate instance
        # variable for clarity
        self._feature_group_thresh = 0

        # pd.DataFrame holding current features
        self._features = pd.DataFrame()

        # dict of generation number to dict dict representation dataframe
        # of features retained at each generation to be emitted
        self._final_features: Dict[int, DataFrameDict] = {}

    def extract_features(self) -> DataFrameLike:
        """
        Perform recursive feature extraction to return DataFrame of features
        """
        # return already calculated features if stored in state
        if self._final_features:
            return self._finalize_features()

        # initialization: generation 0 features are neighborhood features
        features = self.graph.get_neighborhood_features()
        self._update(features)

        for generation in range(1, self.max_generations):

            self.generation_count = generation
            self._feature_group_thresh = generation

            features = self._get_next_features()
            self._update(features)

            # stop if an iteration results in no features retained
            if not self._final_features[generation]:
                break

        return self._finalize_features()

    def _finalize_features(self) -> DataFrameLike:
        """
        Return DataFrame of final features
        """
        all_features_dict = dict(ChainMap(*self._final_features.values()))
        return pd.DataFrame(all_features_dict)

    def _get_next_features(self) -> DataFrameLike:
        """
        Return next level of recursive features (aggregations of node
        features from previous generation)
        """
        # get nodes neighbors and aggregate their previous generation features
        prev_features = self._final_features[self.generation_count - 1].keys()
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
        # add features
        self._features = (
            pd.concat([self._features, features], axis=1, sort=True)
            # fill nans resulting from concatenation where features does not
            # contain neighborless nodes (out-degree=0) on its axis
            .fillna(0)
        )
        # prune redundant features
        pruner = FeaturePruner(self._final_features, self._feature_group_thresh)
        features_to_drop = pruner.prune_features(self._features)
        self._features = self._features.drop(features_to_drop, axis=1)
        # save features that remain after pruning and that
        # have not previously been saved as final features
        retained = features.columns.difference(features_to_drop)
        feature_dict = as_frame(self._features[retained]).to_dict()
        self._final_features[self.generation_count] = feature_dict

    @staticmethod
    def _aggregated_df_to_dict(agg_df: DataFrameLike) -> Dict[str, float]:
        """
        Transform DataFrame of aggregated features to dict formatted for
        concatenation with self._features DataFrame
        :param agg_df: agregated features resulting from df.agg(self.aggs)
        """
        try:
            agg_dicts = agg_df.to_dict(orient='index')
        except TypeError:
            # pd.Series objects do not have to_dict method with orient kwarg
            # so cast to pd.DataFrame and transpose for correct shape
            agg_dicts = agg_df.to_frame().T.to_dict(orient='index')

        formatted_agg_dict = {
            f'{key}({idx})': val
            for idx, row in agg_dicts.items()
            for key, val in row.items()
        }
        return formatted_agg_dict


# Helper functions

def as_frame(df_like: DataFrameLike) -> pd.DataFrame:
    """
    Helper to safely cast a pd.Series to pd.DataFrame without throwing
    an exception if input is already a pd.DataFrame
    :param df_like: pd.Series or pd.DataFrame
    """
    try:
        return df_like.to_frame()
    except AttributeError:
        return df_like
