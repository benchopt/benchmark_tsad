# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


# Implementing a precision metric, if the anomaly is detected in a window,
# we consider it as detected
def soft_precision(y_true, y_pred, detection_range=3):
    """
    If an anomaly is detected detection_range windows before or after
    the true anomaly, we consider it as detected
    e.g. detection_range=3, if anomaly detected at timestamp 5,
    we consider y_pred correct if there is an anomaly at timestamp 2,..., 8
    """
    y_true_windows = np.lib.stride_tricks.sliding_window_view(
        y_true, window_shape=detection_range*2+1, axis=0)
    idxs = y_true_windows[:, detection_range]
    # idx : iterate over y_pred
    # i : iterate over y_true_windows
    # First element of y_true_windows: first detection_range elements of y_pred

    precision = 0
    for i in range(detection_range):
        if y_pred[i] == 1 and np.any(y_true[:detection_range] == 1):
            precision += 1

    for i, idx in enumerate(idxs):
        if y_pred[i] == 1 and np.any(y_true_windows[i] == 1):
            precision += 1

    for i in range(len(y_pred)-detection_range, len(y_pred)):
        if y_pred[i] == 1 and np.any(y_true[detection_range:] == 1):
            precision += 1

    return precision / np.sum(y_pred)
