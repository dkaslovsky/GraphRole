from collections import defaultdict
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from graphrole.types import FactorTuple


MIN_ROLES, MAX_ROLES = (2, 8)
MIN_BITS, MAX_BITS = (1, 8)


def extract_role_factors(
    features: pd.DataFrame,
    n_roles: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper for extracting role factors from a node feature DataFrame
     and returning factor DataFrames
    :param features: DataFrame with rows of node features
    :param n_roles: number of roles to extract or None for automatic selection
    """
    if n_roles:
        # factors will be of shape (n_nodes x n_roles) and (n_roles x n_features) for
        # a total of n_roles * (n_nodes + n_features), so encode with approximately
        # log2(n_roles * (n_nodes + n_features)) bits
        n_bits = int(np.log2(n_roles * sum(features.shape)))
        node_role_ndarray, role_feature_ndarray = get_role_factors(features, n_roles, n_bits)
    else:
        node_role_ndarray, role_feature_ndarray = select_model(features)

    role_labels = [f'role_{i}' for i in range(node_role_ndarray.shape[1])]

    node_role_df = pd.DataFrame(
        node_role_ndarray,
        index=features.index,
        columns=role_labels
    )
    role_feature_df = pd.DataFrame(
        role_feature_ndarray,
        index=role_labels,
        columns=features.columns
    )

    return node_role_df, role_feature_df


def select_model(features: pd.DataFrame) -> FactorTuple:
    """
    Select optimal model via grid search over n_roles and n_bits as measured
     by Minimum Description Length
    :param features: DataFrame with rows of node features
    """
    # define grid
    max_bits_grid_idx = 1 + MAX_BITS
    max_roles_grid_idx = 1 + min(min(features.shape), MAX_ROLES)
    n_bits_grid = range(MIN_BITS, max_bits_grid_idx)
    n_roles_grid = range(MIN_ROLES, max_roles_grid_idx)

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
                model = get_role_factors(features, roles, bits)
                encoding_cost, error_cost = get_description_length(features, model, bits)
            except ValueError:
                # raised when bits is too large to quantize the number of samples
                continue

            encoding_costs[roles, bits] = encoding_cost
            error_costs[roles, bits] = error_cost
            factors[roles][bits] = model

    # select factors with minimal cost
    costs = rescale_costs(encoding_costs) + rescale_costs(error_costs)
    min_cost = np.nanmin(costs)
    # we could catch an IndexError here, but if np.argwhere returns empty there is
    # no way to handle model selection and hence no way to recover
    min_role, min_bits = np.argwhere(costs == min_cost)[0]
    min_model = factors[min_role][min_bits]
    return min_model


def rescale_costs(costs: np.ndarray) -> np.ndarray:
    """
    Rescale the cost matrices for a fixed n_role so that
     encoding and error costs are on the same scale
    :param costs: matrix of costs with n_roles across roles and n_bits across columns
    """
    norms = np.linalg.norm(costs, axis=1)
    norms[np.isnan(norms)] = 1.0
    return costs / norms.reshape(costs.shape[0], 1)


def get_role_factors(
    features: pd.DataFrame,
    n_roles: int,
    n_bits: int
) -> FactorTuple:
    """
    Compute encoded NMF decomposition of feature DataFrame
    :param features: DataFrame with rows of node features
    :param n_roles: number of roles (rank of NMF decomposition)
    :param n_bits: number of bits to use for encoding factor matrices
    """
    n_bins = int(2**n_bits)
    V = features.values
    G, F = get_nmf_factorization(V, n_roles)
    G_encoded = encode(G, n_bins)
    F_encoded = encode(F, n_bins)
    return G_encoded, F_encoded


def get_nmf_factorization(
    X: np.ndarray,
    n_roles: int
) -> FactorTuple:
    """
    Compute NMF factors
    :param X: matrix to factor
    :param n_roles: rank of decomposition
    """
    nmf = NMF(n_components=n_roles, solver='mu')
    with warnings.catch_warnings():
        # ignore convergence warning from NMF since
        # this will result in a large cost anyways
        warnings.simplefilter('ignore')
        G = nmf.fit_transform(X)
        F = nmf.components_
    return G, F


def encode(
    X: np.ndarray,
    n_bins: int
) -> np.ndarray:
    """
    Encode (quantize) a matrix X using a specified number of bins
    :param X: matrix to encode
    :param n_bins: number of bins for encoding
    """
    # quantize using Lloyd-Max quantizier which can be computed using kmeans
    # https://en.wikipedia.org/wiki/Quantization_(signal_processing)
    data = X.reshape(X.size, 1)
    quantizer = KMeans(n_clusters=n_bins)
    with warnings.catch_warnings():
        # ignore convergence warning from kmeans since
        # this will result in a large cost anyways
        warnings.simplefilter('ignore')
        quantizer.fit(data)
    bin_vals = quantizer.cluster_centers_
    quantized = np.array([bin_vals[label] for label in quantizer.labels_])
    return quantized.reshape(X.shape)


def get_description_length(
    features: pd.DataFrame,
    model: Tuple[np.ndarray, np.ndarray],
    n_bits: int
) -> Tuple[float, float]:
    """
    Compute description length for encoding the model tuple (factor matrices)
     using the specified number of bins
    :param features: original DataFrame of features from which factors were computed
    :param model: tuple of encoded NMF factors
    :param n_bits: number of bits used for encoding
    """
    G_encoded, F_encoded = model
    V_approx = G_encoded @ F_encoded
    V = features.values

    encoding_cost = n_bits * (G_encoded.size + F_encoded.size)
    error_cost = get_error_cost(V, V_approx)
    return encoding_cost, error_cost


def get_error_cost(
    V: np.ndarray,
    V_approx: np.ndarray
) -> float:
    """
    Compute error cost of encoding for description length
    :param V: original matrix
    :param V_approx: reconstructed matrix from encoded factors
    """
    # KL divergence as given in section 2.3 of RolX paper
    vec1 = V.ravel()
    vec2 = V_approx.ravel()
    kl_div = np.sum(np.where(vec1 != 0, vec1 * np.log(vec1 / vec2) - vec1 + vec2, 0))
    return kl_div
