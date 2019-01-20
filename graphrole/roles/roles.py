from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


MIN_ROLES, MAX_ROLES = 2, 8
MIN_BITS, MAX_BITS = 1, 8


def extract_role_factors(
    features: pd.DataFrame,
    n_roles: Optional[int] = None,
    verbose: Optional[bool] = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
    n_roles: int,
    verbose: bool
) -> Tuple[np.ndarray, np.ndarray]:

    if n_roles:
        # default n_bins to 2**np.log2(np.log2(features.shape[0])) = np.log2(features.shape[0])
        n_bits = np.log2(np.log2(features.shape[0]))
        return get_role_factors(features, n_roles, n_bits)

    # model selection
    max_roles = min(min(features.shape), MAX_ROLES)
    n_roles_grid = range(2, max_roles + 1)
    n_bits_grid = range(1, MAX_BITS + 1)

    min_cost = np.inf
    min_code = (None, None)

    for roles in n_roles_grid:
        for bits in n_bits_grid:

            try:
                code = get_role_factors(features, roles, bits)
                cost, *components = get_description_length(features, code, bits)
            except ValueError:
                # raised when bits is too large to quantize the number of samples
                cost = np.inf

            if verbose:
                report_cost(roles, bits, cost, components)

            if cost < min_cost:
                min_cost = cost
                min_code = code

    if verbose:
        report_model(min_code)

    return min_code


def get_role_factors(
    features: pd.DataFrame,
    n_roles: int,
    n_bits: int
) -> Tuple[np.ndarray, np.ndarray]:

    n_bins = int(2**n_bits)
    V = features.values
    G, F = get_nmf_factors(V, n_roles)
    G_encoded = encode(G, n_bins)
    F_encoded = encode(F, n_bins)
    return G_encoded, F_encoded


def get_nmf_factors(X: np.ndarray, n_roles: int) -> Tuple[np.ndarray, np.ndarray]:
    nmf = NMF(n_components=n_roles, solver='mu')
    with warnings.catch_warnings():
        # ignore convergence warning from NMF since
        # this will result in a large cost anyways
        warnings.simplefilter('ignore')
        G = nmf.fit_transform(X)
        F = nmf.components_
    return G, F


def encode(X: np.ndarray, n_bins: int) -> np.ndarray:
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

    G_encoded, F_encoded = code
    V_approx = G_encoded @ F_encoded
    V = features.values

    encoding_cost = n_bits * (G_encoded.size + F_encoded.size)
    error_cost = get_error_cost(V, V_approx)
    total_cost = encoding_cost + error_cost
    return total_cost, encoding_cost, error_cost


def get_error_cost(V: np.ndarray, V_approx: np.ndarray) -> float:
    # KL divergence as given in RolX paper
    vec1 = V.ravel()
    vec2 = V_approx.ravel()
    kl_div = np.sum(np.where(vec1 != 0, vec1 * np.log(vec1 / vec2) - vec1 + vec2, 0))
    return kl_div


def report_cost(roles: int, bits: int, cost: float, components: Tuple[float, float]) -> None:
    info = f'roles={roles}, bits={bits}: cost={cost:.2f}'
    if cost < np.inf:
        encoding_cost, error_cost = components
        info += f' (encoding={encoding_cost:.2f}, error={error_cost:.2f})'
    print(info)


def report_model(code: Tuple[float, float]) -> None:
    G, _ = code
    roles = G.shape[1]
    print(f'Model with {roles} roles selected')
