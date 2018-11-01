import itertools as it

import pandas as pd

from graphrole.features.similarity import group_features, vertical_log_binning
from graphrole.graph import Graph, NetworkxGraph


class RecursiveFeatureExtractor:

    recursive_aggs = [
        pd.DataFrame.sum,
        pd.DataFrame.mean,
    ]

    def __init__(self,
                 G,
                 max_generations: int = 10):
        
        self.graph = NetworkxGraph(G)
        self.max_generations = max_generations
        
        self.dist_thresh = 0
        self.generation_count = 0

        self.generation_dict = {}
        
        self.features = None
        self.binned_features = None

        self.final_features = []
        self.final_features_names = set()

    def extract_features(self):
        for generation in range(self.max_generations):
            
            self.generation_count = generation
            self.dist_thresh = generation 

            next_features = self._get_next_features()
            self._update(next_features)
            # stop if a recursive iteration 
            # results in no features retained
            retained_features = (
                self.generation_dict[generation]
                .intersection(self.features.columns)
            )
            if not retained_features:
                return self._finalize_features()
        
        return self._finalize_features()
    
    def _finalize_features(self):
        return pd.concat(self.final_features, axis=1)

    def _get_next_features(self):
        # initial (generation 0) features are neighborhood features
        if self.generation_count == 0:
            return self.graph.get_neighborhood_features()
        # get nodes neighbors and aggregate
        # their previous generation features
        prev_features = self.generation_dict[self.generation_count - 1]
        features = {
            node: (
                self.features
                .reindex(index=self.graph.get_neighbors(node), columns=prev_features)
                .agg(self.recursive_aggs)
                .pipe(self._aggregated_df_to_dict)
            )
            for node in self.graph.get_nodes()
        }
        return pd.DataFrame.from_dict(features, orient='index')

    def _update(self, features):
        self._add_features(features)
        self._prune_features()
        # save features that remain after pruning 
        # and that have not previously been saved
        new_features = (
            self.features.columns
            .difference(self.final_features_names)
        )
        self.final_features.append(self.features[new_features])
        self.final_features_names.update(new_features)

    def _prune_features(self):
        features_to_drop = []
        groups = group_features(self.binned_features, dist_thresh=self.dist_thresh)
        for group in groups:
            oldest = self._get_oldest_feature(group)
            to_drop = group - {oldest}
            features_to_drop.extend(to_drop)
        self._drop_features(features_to_drop)

    # TODO: how to break ties in reproducible way?
    def _get_oldest_feature(self, features: set):
        for gen in range(self.generation_count):
            cur_gen = features.intersection(self.generation_dict[gen])
            if cur_gen:
                return next(iter(cur_gen))
        return next(iter(features))
    
    def _add_features(self, features):
        binned_features = features.apply(vertical_log_binning)
        self.features = pd.concat([self.features, features], axis=1)
        self.binned_features = pd.concat([self.binned_features, binned_features], axis=1)
        self.generation_dict[self.generation_count] = set(features.columns)
    
    def _drop_features(self, features):
        self.features.drop(features, axis=1, inplace=True)
        self.binned_features.drop(features, axis=1, inplace=True)
        
    @staticmethod
    def _aggregated_df_to_dict(agg_df: pd.DataFrame):
        if isinstance(agg_df, pd.Series):
            agg_df = agg_df.to_frame()
        agg_dicts = agg_df.to_dict(orient='index')
        return {
            f'{key}_{idx}': val
            for idx, row in agg_dicts.items()
            for key, val in row.items()
        }
