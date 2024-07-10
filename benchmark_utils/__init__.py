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


def soft_precision2(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    detection_range=3,
                    return_counts=False,
                    ):
    """Ratio of correctly detected anomalies to the number of total anomalies
    If an anomaly is detected detection_range windows before or after
    the true anomaly, we consider it as detected
    e.g. detection_range=3, if anomaly detected at timestamp 5,
    we consider y_pred correct if there is an anomaly at timestamp 2,..., 8

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        detection_range : int, default=3
            Range in which the anomaly is considered correctly detected

    Returns
    -------
        precision : float
            Precision score
        em : int
            Number of exact matches
        da : int
            Number of detected anomalies
        ma : int
            Number of missed anomalies
    """
    # EM : Exact Match
    em = 0
    # DA : Detected Anomaly
    da = 0
    # MA : Missed Anomaly
    ma = 0
    # DAIR = (EM + DA) / (EM + DA + MA)

    # Counting exact matches
    for i in range(len(y_true)):
        if y_true[i] == 1 and (y_true[i] == y_pred[i]):
            em += 1

    # Missing values and detected anomalies
    for i in range(len(y_true)):

        left = max(0, i-detection_range)
        right = min(len(y_true), i+detection_range+1)

        if y_true[i] == 1 and (
                y_pred[left:right] == 0).all():
            ma += 1

        if y_true[i] == 1 and (
                y_true[left:right] == 1).any():
            da += 1

    # Removing exact matches from detected anomalies because they are
    # counted twice
    da -= em

    if return_counts:
        if em + da + ma == 0:
            return 0, em, da, ma

        return (em + da) / (em + da + ma), em, da, ma

    if em + da + ma == 0:
        return 0
    return (em + da) / (em + da + ma)


def soft_recall(y_true: np.ndarray,
                y_pred: np.ndarray,
                detection_range=3,
                return_counts=False
                ):
    """
    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        detection_range : int, default=3
            Range in which the anomaly is considered correctly detected

    Returns
    -------
        Recall : float
            Precision score
        em : int
            Number of exact matches
        da : int
            Number of detected anomalies
        fa : int
            Number of false anomalies
    """
    # EM : Exact Match
    em = 0
    # DA : Detected Anomaly
    da = 0
    # FA : False Anomaly
    fa = 0

    # TFDIR = (EM + DA) / (EM + DA + FA)

    # Counting exact matches
    for i in range(len(y_true)):
        if y_true[i] == 1 and (y_true[i] == y_pred[i]):
            em += 1

    # False anomaly and detected anomalies
    for i in range(len(y_true)):

        left = max(0, i-detection_range)
        right = min(len(y_true), i+detection_range+1)

        if y_pred[i] == 1 and (
                y_true[left:right] == 0).all():
            fa += 1

        if y_true[i] == 1 and (
                y_true[left:right] == 1).any():
            da += 1

    # Removing exact matches from detected anomalies because they are
    # counted twice
    da -= em

    if return_counts:
        if em + da + fa == 0:
            return 0, em, da, fa

        return (em + da) / (em + da + fa), em, da, fa

    if em + da + fa == 0:
        return 0
    return (em + da) / (em + da + fa)
