import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from graphrole.types import FactorTuple


def get_nmf_decomposition(
    X: np.ndarray,
    n_roles: int,
) -> FactorTuple:
    """
    Compute NMF decomposition
    :param X: matrix to factor
    :param n_roles: rank of decomposition
    """
    nmf = NMF(n_components=n_roles, solver='mu', init='nndsvda')
    with warnings.catch_warnings():
        # ignore convergence warning from NMF since
        # this will result in a large cost anyways
        warnings.simplefilter('ignore')
        G = nmf.fit_transform(X)
        F = nmf.components_
    return G, F


def encode(
    X: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """
    Encode (quantize) a matrix X using a specified number of bins
    :param X: matrix to encode
    :param n_bins: number of bins for encoding
    """
    # quantize using Lloyd-Max quantizier which can be computed using kmeans
    # https://en.wikipedia.org/wiki/Quantization_(signal_processing)
    data = X.reshape(X.size, 1)
    quantizer = KMeans(n_clusters=n_bins, random_state=1)
    with warnings.catch_warnings():
        # ignore convergence warning from kmeans since
        # this will result in a large cost anyways
        warnings.simplefilter('ignore')
        quantizer.fit(data)
    bin_vals = quantizer.cluster_centers_
    quantized = np.array([bin_vals[label] for label in quantizer.labels_])
    return quantized.reshape(X.shape)
