from typing import Union

import numpy as np
import pandas as pd

VectorLike = Union[np.array, pd.Series]


def vertical_log_binning(arr: VectorLike, frac: float = 0.5) -> VectorLike:
    """
    Reassigns values of an array into vertical logarithmic bins
    :param arr: array to be binned
    :param frac: value in (0, 1) defining fraction of values assigned to each bin
    """

    if not 0 < frac < 1:
        raise ValueError('must specify frac in interval (0, 1)')

    arr_len = len(arr)
    binned = np.zeros(arr_len, dtype=np.int)
    
    # get sorted unique values and counts in arr
    arr_uniq, counts = np.unique(arr, return_counts=True)
    # convert to cumulative counts
    counts = np.cumsum(counts)

    # initial iteration parameters
    binned_len = 0                  # length of binned portion of arr
    unbinned_len = arr_len          # length of unbinned portion of arr
    bin_min = -np.inf               # left side value of current bin (exclusive)

    for bin_val in range(arr_len):
        
        # bin size is fraction frac of the unbinned len (enforce at least 1)
        bin_size = max(int(frac * unbinned_len), 1)
        # get index of largest unique value from arr to be included in bin
        u_idx = np.searchsorted(counts, binned_len + bin_size)
        bin_max = arr_uniq[u_idx]
        # mark members of current bin with bin_val
        arr_idx = np.logical_and(arr > bin_min, arr <= bin_max)
        binned[arr_idx] = bin_val

        # update iteration paramters
        binned_len += sum(arr_idx)
        unbinned_len = arr_len - binned_len
        bin_min = bin_max
        
        # check if all values have been binned
        if unbinned_len == 0:
            break

    return binned
