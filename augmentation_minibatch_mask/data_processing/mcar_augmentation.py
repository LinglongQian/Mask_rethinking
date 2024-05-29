"""
Corrupt data by adding missing values to it with MCAR (missing completely at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union
import math
import numpy as np
import torch

def _mcar_numpy(
    X: np.ndarray,
    p: float,
) -> np.ndarray:
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    observed_masks = ~np.isnan(X)
    mcar_missing_mask = np.copy(observed_masks.reshape(-1))
    obs_indices = np.where(mcar_missing_mask)[0]
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * p), replace=False)
    mcar_missing_mask[miss_indices] = False
    mcar_missing_mask = mcar_missing_mask.reshape(X.shape)
    X[~mcar_missing_mask] = np.nan  # mask values selected by mcar_missing_mask
    return X


def _mcar_torch(
    X: torch.Tensor,
    p: float,
) -> torch.Tensor:
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    observed_masks = ~torch.isnan(X)
    mcar_missing_mask = observed_masks.view(-1).clone()
    obs_indices = torch.where(mcar_missing_mask)[0]
    miss_indices = torch.tensor(np.random.choice(obs_indices.numpy(), int(len(obs_indices) * p), replace=False))
    mcar_missing_mask[miss_indices] = False
    mcar_missing_mask = mcar_missing_mask.view(X.shape)
    X[~mcar_missing_mask] = torch.nan  # mask values selected by mcar_missing_mask
    return X

def mcar_augmentation(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create completely random missing values (MCAR case).

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p : float, in (0,1),
        The probability that values may be masked as missing completely at random.
        Note that the values are randomly selected no matter if they are originally missing or observed.
        If the selected values are originally missing, they will be kept as missing.
        If the selected values are originally observed, they will be masked as missing.
        Therefore, if the given X already contains missing data, the final missing rate in the output X could be
        in range [original_missing_rate, original_missing_rate+rate], but not strictly equal to
        `original_missing_rate+rate`. Because the selected values to be artificially masked out may be originally
        missing, and the masking operation on the values will do nothing.

    Returns
    -------
    corrupted_X : array-like
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mcar_numpy(X, p)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mcar_torch(X, p)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
