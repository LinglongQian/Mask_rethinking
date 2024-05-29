"""
Corrupt data by adding missing values to it with MCAR (missing completely at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union
import math
import numpy as np
import torch


def adjust_probability_vectorized_numpy(obs_count, avg_count, base_prob, increase_factor=0.5):
    adjusted_prob = np.where(
        obs_count < avg_count,
        np.minimum(base_prob * (avg_count / obs_count) * increase_factor, 1.0),
        np.maximum(base_prob * (obs_count / avg_count) / increase_factor, 0)
    )
    return adjusted_prob

def adjust_probability_vectorized_torch(obs_count, avg_count, base_prob, increase_factor=0.5):
    adjusted_prob = torch.where(
        obs_count < avg_count,
        torch.minimum(base_prob * (avg_count / obs_count) * increase_factor, torch.tensor(1.0)),
        torch.maximum(base_prob * (obs_count / avg_count) / increase_factor, torch.tensor(0.0))
    )
    return adjusted_prob

# def non_uniform_masking(data, p, pre_replacement_probabilities=None, increase_factor=0.1):
#     # Get Dimensionality
#     [N, T, D] = data.shape

#     if pre_replacement_probabilities is None:

#         observations_per_feature = np.sum(~np.isnan(data), axis=(0, 1))
#         average_observations = np.mean(observations_per_feature)
#         replacement_probabilities = np.full(D, p / 100)

#         if increase_factor > 0:
#             print('The increase_factor is {}!'.format(increase_factor))
#             for feature_idx in range(D):
#                 replacement_probabilities[feature_idx] = adjust_probability_vectorized(
#                     observations_per_feature[feature_idx],
#                     average_observations,
#                     replacement_probabilities[feature_idx],
#                     increase_factor=increase_factor
#                 )
            
#             # print('before:\n',replacement_probabilities)
#             total_observations = np.sum(observations_per_feature)
#             total_replacement_target = total_observations * p / 100

#             for _ in range(1000):  # Limit iterations to prevent infinite loop
#                 total_replacement = np.sum(replacement_probabilities * observations_per_feature)
#                 if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
#                     break
#                 adjustment_factor = total_replacement_target / total_replacement
#                 replacement_probabilities *= adjustment_factor
            
#             # print('after:\n',replacement_probabilities)
#     else:
#         replacement_probabilities = pre_replacement_probabilities

#     values = copy.deepcopy(data)
#     random_matrix = np.random.rand(N, T, D)
#     if p > 0:
#         values[(~np.isnan(values)) & (random_matrix < replacement_probabilities)] = np.nan
#     return values, replacement_probabilities
    
def _non_uniform_masking_numpy(
    X: np.ndarray,
    p: float,
    pre_replacement_probabilities=None,
    increase_factor=0.1,
) -> (np.ndarray, np.ndarray):
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    
    N, T, D = X.shape

    if pre_replacement_probabilities is None:
        observations_per_feature = np.sum(~np.isnan(X), axis=(0, 1))
        average_observations = np.mean(observations_per_feature)
        replacement_probabilities = np.full(D, p)

        if increase_factor > 0:
            replacement_probabilities = adjust_probability_vectorized_numpy(
                observations_per_feature, average_observations, replacement_probabilities, increase_factor
            )
            # for feature_idx in range(D):
            #     replacement_probabilities[feature_idx] = adjust_probability_vectorized(
            #         observations_per_feature[feature_idx],
            #         average_observations,
            #         replacement_probabilities[feature_idx],
            #         increase_factor=increase_factor
            #     )
            
            total_observations = np.sum(observations_per_feature)
            total_replacement_target = total_observations * p

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = np.sum(replacement_probabilities * observations_per_feature)
                if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
    else:
        replacement_probabilities = pre_replacement_probabilities

    values = np.copy(X)
    random_matrix = np.random.rand(N, T, D)
    if p > 0:
        mask = (~np.isnan(values)) & (random_matrix < replacement_probabilities)
        values[mask] = np.nan
    return values, replacement_probabilities


def _non_uniform_masking_torch(
    X: torch.Tensor,
    p: float,
    pre_replacement_probabilities=None,
    increase_factor=0.1,
) -> (torch.Tensor, torch.Tensor):
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    
    N, T, D = X.shape

    if pre_replacement_probabilities is None:
        observations_per_feature = torch.sum(~torch.isnan(X), dim=(0, 1)).float()
        average_observations = torch.mean(observations_per_feature)
        replacement_probabilities = torch.full((D,), p)

        if increase_factor > 0:
            replacement_probabilities = adjust_probability_vectorized_torch(
                observations_per_feature, average_observations, replacement_probabilities, increase_factor
            )
            # for feature_idx in range(D):
            #     replacement_probabilities[feature_idx] = adjust_probability_vectorized(
            #         observations_per_feature[feature_idx].item(),
            #         average_observations.item(),
            #         replacement_probabilities[feature_idx].item(),
            #         increase_factor=increase_factor
            #     )
            
            total_observations = torch.sum(observations_per_feature)
            total_replacement_target = total_observations * p

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = torch.sum(replacement_probabilities * observations_per_feature)
                if torch.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
    else:
        replacement_probabilities = pre_replacement_probabilities

    values = torch.clone(X)
    random_matrix = torch.rand(N, T, D)
    if p > 0:
        mask = (~torch.isnan(values)) & (random_matrix < replacement_probabilities)
        values[mask] = torch.nan
    return values, replacement_probabilities

def non_uniform_masking(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    pre_replacement_probabilities=None,
    increase_factor=0.1,
) -> Union[np.ndarray, torch.Tensor]:
    """Create completely random missing values (MCAR case).

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p : float, in [0,1],
        The probability that values may be masked as missing completely at random.
        Note that the values are randomly selected no matter if they are originally missing or observed.
        If the selected values are originally missing, they will be kept as missing.
        If the selected values are originally observed, they will be masked as missing.
        Therefore, if the given X already contains missing data, the final missing rate in the output X could be
        in range [original_missing_rate, original_missing_rate+rate], but not strictly equal to
        `original_missing_rate+rate`. Because the selected values to be artificially masked out may be originally
        missing, and the masking operation on the values will do nothing.

    pre_replacement_probabilities : array-like, optional
        Pre-calculated replacement probabilities for each feature. If provided, these will be used instead of calculating new ones.
    
    increase_factor : float, optional, default=0.1
        Factor to increase/decrease the masking probability for features with fewer/more observations.

    Returns
    -------
    corrupted_X : array-like
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    replacement_probabilities : array-like
        The replacement probabilities used for masking each feature.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X, replacement_probabilities = _non_uniform_masking_numpy(X, p, pre_replacement_probabilities, increase_factor)
    elif isinstance(X, torch.Tensor):
        corrupted_X, replacement_probabilities = _non_uniform_masking_torch(X, p, pre_replacement_probabilities, increase_factor)
    else:
        raise TypeError(
            "X must be of type list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X, replacement_probabilities
