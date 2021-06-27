from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from graphrole.roles.description_length import get_description_length_costs
from graphrole.roles.factor import encode, get_nmf_decomposition
from graphrole.types import DataFrameLike, FactorTuple, Node


class RoleExtractor:

    """ Assign node roles based on input features """

    N_ROLE_RANGE = (2, 8)
    N_BIT_RANGE = (1, 8)

    def __init__(
        self,
        n_roles: Optional[int] = None,
        n_role_range: Optional[Tuple[int, int]] = None,
        n_bit_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        :param n_roles: optional number of roles to select; default uses MDL model selection
        :param n_role_range: optional tuple for (min, max) roles for model selection grid search
        :param n_bit_range: optional tuple for (min, max) bits for model selection grid search
        """
        self.n_roles = n_roles

        self.min_roles, self.max_roles = n_role_range if n_role_range else self.N_ROLE_RANGE
        self.min_bits, self.max_bits = n_bit_range if n_bit_range else self.N_BIT_RANGE

        self.node_role_factor: Optional[pd.DataFrame] = None
        self.role_feature_factor: Optional[pd.DataFrame] = None

    @property
    def roles(self) -> Optional[Dict[Node, float]]:
        """
        Return dict mapping node to role
        """
        try:
            role_df = self.node_role_factor.idxmax(axis=1)
            return role_df.to_dict()
        except AttributeError:
            return None

    @property
    def role_percentage(self) -> Optional[DataFrameLike]:
        """
        Return DataFrameLike with percent role (columns) for each node (index)
        """
        try:
            return self.node_role_factor.apply(lambda row: row / row.sum(), axis=1)
        except AttributeError:
            return None

    def extract_role_factors(
        self,
        features: pd.DataFrame,
    ) -> None:
        """
        Wrapper for extracting role factors from a node feature DataFrame
        and returning factor DataFrames
        :param features: DataFrame with rows of node features
        """
        if self.n_roles:
            # factors will be of shape (n_nodes x n_roles) and (n_roles x n_features) for
            # a total of n_roles * (n_nodes + n_features), so encode with approximately
            # log2(n_roles * (n_nodes + n_features)) bits
            n_bits = int(np.log2(self.n_roles * min(features.shape)))
            node_role_ndarray, role_feature_ndarray = self._get_encoded_role_factors(
                features, self.n_roles, n_bits
            )
        else:
            node_role_ndarray, role_feature_ndarray = self._select_model(features)

        role_labels = [
            f'role_{i}'
            for i in range(node_role_ndarray.shape[1])
        ]

        self.node_role_factor = pd.DataFrame(
            node_role_ndarray,
            index=features.index,
            columns=role_labels
        )
        self.role_feature_factor = pd.DataFrame(
            role_feature_ndarray,
            index=role_labels,
            columns=features.columns
        )

    def explain(self):
        raise NotImplementedError('Role explanation (\"sense making\") is not yet implemented.')

    def _select_model(
        self,
        features: pd.DataFrame,
    ) -> FactorTuple:
        """
        Select optimal model via grid search over n_roles and n_bits as measured
        by Minimum Description Length
        :param features: DataFrame with rows of node features
        """
        # define grid
        max_bits_grid_idx = self.max_bits + 1
        max_roles_grid_idx = min(min(features.shape), self.max_roles) + 1
        n_bits_grid = range(self.min_bits, max_bits_grid_idx)
        n_roles_grid = range(self.min_roles, max_roles_grid_idx)

        # ndarrays to store encoding and error costs
        matrix_dims = (max_roles_grid_idx, max_bits_grid_idx)
        encoding_costs = np.full(matrix_dims, np.nan)
        error_costs = np.full(matrix_dims, np.nan)
        # dict to store factor tuples
        factors = defaultdict(dict)

        # grid search
        for roles in n_roles_grid:
            for bits in n_bits_grid:

                try:
                    model = self._get_encoded_role_factors(features, roles, bits)
                    encoding_cost, error_cost = get_description_length_costs(features, model)
                except ValueError:
                    # raised when bits is too large to quantize the number of samples
                    continue

                encoding_costs[roles, bits] = encoding_cost
                error_costs[roles, bits] = error_cost
                factors[roles][bits] = model

        # select factors with minimal cost
        costs = self._rescale_costs(encoding_costs) + self._rescale_costs(error_costs)
        min_cost = np.nanmin(costs)
        # we could catch an IndexError here, but if np.argwhere returns empty there is
        # no way to handle model selection and hence no way to recover
        min_role, min_bits = np.argwhere(costs == min_cost)[0]
        min_model = factors[min_role][min_bits]
        return min_model

    @staticmethod
    def _get_encoded_role_factors(
        features: pd.DataFrame,
        n_roles: int,
        n_bits: int,
    ) -> FactorTuple:
        """
        Compute encoded NMF decomposition of feature DataFrame
        :param features: DataFrame with rows of node features
        :param n_roles: number of roles (rank of NMF decomposition)
        :param n_bits: number of bits to use for encoding factor matrices
        """
        n_bins = int(2**n_bits)
        V = features.values
        G, F = get_nmf_decomposition(V, n_roles)
        G_encoded = encode(G, n_bins)
        F_encoded = encode(F, n_bins)
        return G_encoded, F_encoded

    @staticmethod
    def _rescale_costs(
        costs: np.ndarray
    ) -> np.ndarray:
        """
        Rescale the cost matrices for a fixed n_role so that
        encoding and error costs are on the same scale
        :param costs: matrix of costs with n_roles across roles and n_bits across columns
        """
        norms_skip_nan = np.sqrt(np.nansum(np.square(costs), axis=1))
        return costs / norms_skip_nan.reshape(costs.shape[0], 1)
