from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


def extract_role_factors(
    features: pd.DataFrame,
    n_roles: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return node-role and role-feature factor DataFrames
    :param features: DataFrame of features
    :param n_roles: optional number of roles to extract
    """
    n_roles = n_roles if n_roles is not None else get_num_roles(features)

    nmf = NMF(n_components=n_roles, solver='mu')
    role_labels = [f'role_{i}' for i in range(n_roles)]

    node_role_array = nmf.fit_transform(features)
    role_feature_array = nmf.components_

    node_role_df = pd.DataFrame(
        node_role_array,
        index=features.index,
        columns=role_labels
    )
    role_feature_df = pd.DataFrame(
        role_feature_array,
        index=role_labels,
        columns=features.columns
    )
    return node_role_df, role_feature_df


def get_num_roles(features: pd.DataFrame) -> int:
    """
    Return estimate of number of node roles present based on node features
    :param features: DataFrame of features
    """
    return 2


# TODO: need to normalize encoding cost / error cost to the same scale?
def description_len(nmf_tuple: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], bits: int) -> float:

    (V, G, F) = nmf_tuple

    bins = 2**bits
    G_quantized = quantize(G.values, bins)
    F_quantized = quantize(F.values, bins)
    V_encoded = G_quantized @ F_quantized

    encoding_cost = bits * (G_quantized.size + F_quantized.size)
    error_cost = kl_divergence(V.values.ravel(), V_encoded.ravel())  # TODO: eliminate ravel
    print(encoding_cost, error_cost)
    return encoding_cost + error_cost


def quantize(X: np.ndarray, bins: int) -> np.array:
    # quantize using Lloyd-Max quantizier which can be computed using kmeans
    # https://en.wikipedia.org/wiki/Quantization_(signal_processing)
    data = X.reshape(X.size, 1)
    quantizer = KMeans(n_clusters=bins).fit(data)
    bin_vals = quantizer.cluster_centers_
    quantized = np.array([bin_vals[label] for label in quantizer.labels_])
    return quantized.reshape(X.shape)


# TODO: check if this is equal to https://scipy.github.io/devdocs/generated/scipy.stats.entropy.html
# TODO: check if this is the same as the equation given in the paper
def kl_divergence(v1, v2):
    return entropy(pk=v1, qk=v2, base=2)
