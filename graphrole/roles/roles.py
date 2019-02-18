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
    n_roles: Optional[int] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper for extracting role factors from a node feature DataFrame
     and returning factor DataFrames
    :param features: DataFrame with rows of node features
    :param n_roles: number of roles to extract or None for automatic selection
    :param verbose: flag for printing model cost at each iteration
    """

    node_role_ndarray, role_feature_ndarray = get_role_ndarrays(features, n_roles, verbose)

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


def get_role_ndarrays(
    features: pd.DataFrame,
    n_roles: Optional[int],
    verbose: bool
) -> FactorTuple:
    """
    Main function for extracting role factors; model selection is automated using
     Minimum Description Length
    :param features: DataFrame with rows of node features
    :param n_roles: number of roles to extract or None for automatic selection
    :param verbose: flag for printing model cost at each iteration
    """

    # model selection
    if n_roles:
        n_roles_grid = [n_roles]
    else:
        max_roles = min(min(features.shape), MAX_ROLES)
        n_roles_grid = range(MIN_ROLES, max_roles + 1)
    n_bits_grid = range(MIN_BITS, MAX_BITS + 1)

    min_cost = np.inf
    min_code = (None, None)

    for roles in n_roles_grid:
        for bits in n_bits_grid:

            try:
                code = get_role_factors(features, roles, bits)
                cost, *components = get_description_length(features, code, bits)
            except ValueError:
                # raised when bits is too large to quantize the number of samples
                cost = np.nan

            if verbose:
                report_cost(roles, bits, cost, components)

            if cost < min_cost:
                min_cost = cost
                min_code = code

    return min_code


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
    G, F = get_nmf_factors(V, n_roles)
    G_encoded = encode(G, n_bins)
    F_encoded = encode(F, n_bins)
    return G_encoded, F_encoded


def get_nmf_factors(
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
    code: Tuple[np.ndarray, np.ndarray],
    n_bits: int
) -> Tuple[float, float, float]:
    """
    Compute description length for encoding the code tuple (factor matrices)
     using the specified number of bins
    :param features: original DataFrame of features from which factors were computed
    :param code: tuple of encoded NMF factors
    :param n_bits: number of bits used for encoding
    """

    G_encoded, F_encoded = code
    V_approx = G_encoded @ F_encoded
    V = features.values

    encoding_cost = n_bits * (G_encoded.size + F_encoded.size)
    error_cost = get_error_cost(V, V_approx)
    total_cost = encoding_cost + error_cost
    return (total_cost, encoding_cost, error_cost)


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


def report_cost(
    roles: int,
    bits: int, 
    cost: float, 
    cost_components: Tuple[float, float]
) -> None:
    """
    Helper for reporting description length cost while iterating to find minimum
    :param roles: number of roles used (rank of NMF decomposition)
    :param bits: number of bits used to encode NMF factors
    :param cost: cost of the encoding (description length)
    :param cost_components: encoding and error cost components of description length
    """
    info = f'roles={roles}, bits={bits}: cost={cost:.2f}'
    #if cost < np.inf:
    if not np.isnan(cost):
        encoding_cost, error_cost = cost_components
        info += f' (encoding={encoding_cost:.2f}, error={error_cost:.2f})'
    print(info)
