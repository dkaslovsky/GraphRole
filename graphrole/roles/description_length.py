from typing import Tuple

import numpy as np

from graphrole.types import FactorTuple, MatrixLike


def get_description_length_costs(
    V: MatrixLike,
    model: FactorTuple,
) -> Tuple[float, float]:
    """
    Compute description length for encoding the model tuple (factor matrices)
     using the specified number of bits
    :param V: original matrix of features from which factors were computed
    :param model: tuple of encoded NMF factors
    """
    G_encoded, F_encoded = model
    V_approx = G_encoded @ F_encoded
    try:
        V_orig = V.values
    except AttributeError:
        # V was already np.ndarray
        V_orig = V

    return (
        get_encoding_cost(model),
        get_error_cost(V_orig, V_approx)
    )


def get_encoding_cost(
    model: FactorTuple,
) -> float:
    G_encoded, F_encoded = model
    # estimate encoding bits
    G_vals = len(np.unique(G_encoded))
    F_vals = len(np.unique(F_encoded))
    n_bins = max(G_vals, F_vals)
    n_bits = np.ceil(np.log2(n_bins))
    return n_bits * (G_encoded.size + F_encoded.size)


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

    # use mask in np.log to avoid runtime warning for zeros and in np.where to ensure correct value
    mask = vec1 != 0
    log_vals = np.log(vec1 / vec2, where=mask, out=np.zeros(vec1.shape))
    kl_div = np.sum(np.where(mask, vec1 * log_vals - vec1 + vec2, 0))
    return kl_div
