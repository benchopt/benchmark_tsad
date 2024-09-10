# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


def mean_overlaping_pred(predictions, stride):
    """
    Averages overlapping predictions for multivariate time series.

    Args:
    predictions: np.ndarray, shape (n_windows, H, n_features)
                The predicted values for each window for each feature.
    stride: int
            The stride size.

    Returns:
    np.ndarray: Averaged predictions for each feature.
    """
    n_windows, H, n_features = predictions.shape
    total_length = (n_windows-1) * stride + H - 1

    # Array to store accumulated predictions for each feature
    accumulated = np.zeros((total_length, n_features))
    # store the count of predictions at each point for each feature
    counts = np.zeros((total_length, n_features))

    # Accumulate predictions and counts based on stride
    for i in range(n_windows):
        start = i * stride
        accumulated[start:start+H] += predictions[i]
        counts[start:start+H] += 1

    # Avoid division by zero
    counts[counts == 0] = 1

    # Average the accumulated predictions
    averaged_predictions = accumulated / counts

    return averaged_predictions
