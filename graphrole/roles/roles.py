from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


def extract_role_factors(features, n_roles=None):
    node_role_ndarray, role_feature_ndarray = get_role_ndarrays(features, n_roles)

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


def get_role_ndarrays(features, n_roles=None):

    if n_roles:
        n_bins = np.log2(features.shape[0])
        return get_role_factors(features, n_roles, n_bins)

    # TODO: error check against features.shape
    # TODO: populate with real values instead of these placeholders
    n_bins_grid = range(2, features.shape[0])
    n_roles_grid = range(2, min(min(features.shape), 8))

    min_cost = np.inf
    min_code = (None, None)

    for n_bins in n_bins_grid:
        for n_roles in n_roles_grid:

            code = get_role_factors(features, n_roles, n_bins)
            cost = get_description_length(features, code, n_bins)

            if cost < min_cost:
                min_cost = cost
                min_code = code

    return min_code


def get_role_factors(features, n_roles, n_bins):
    V = features.values
    G, F = get_nmf_factors(V, n_roles)
    G_encoded = encode(G, n_bins)
    F_encoded = encode(F, n_bins)
    return G_encoded, F_encoded


def get_nmf_factors(X: np.ndarray, n_roles: int) -> Tuple[np.ndarray, np.ndarray]:
    nmf = NMF(n_components=n_roles, solver='mu')
    G = nmf.fit_transform(X)
    F = nmf.components_
    return G, F


def encode(X: np.ndarray, bins: int) -> np.array:
    # quantize using Lloyd-Max quantizier which can be computed using kmeans
    # https://en.wikipedia.org/wiki/Quantization_(signal_processing)
    data = X.reshape(X.size, 1)
    quantizer = KMeans(n_clusters=bins).fit(data)
    bin_vals = quantizer.cluster_centers_
    quantized = np.array([bin_vals[label] for label in quantizer.labels_])
    return quantized.reshape(X.shape)


def get_description_length(features, code, n_bins):
    G_encoded, F_encoded = code
    V_approx = G_encoded @ F_encoded
    V = features.values

    bits = 2**n_bins
    encoding_cost = bits * (G_encoded.size + F_encoded.size)
    error_cost = get_error_cost(V, V_approx)
    print(encoding_cost, error_cost)
    total_cost = encoding_cost + error_cost
    return total_cost


def get_error_cost(V, V_approx):
    # KL divergence as given in paper
    vec1 = V.ravel()
    vec2 = V_approx.ravel()
    kl_div = np.sum(np.where(vec1 != 0, vec1 * np.log(vec1 / vec2) - vec1 + vec2, 0))
    return kl_div
