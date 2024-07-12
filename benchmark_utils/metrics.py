from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


def soft_precision(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   detection_range=3,
                   return_counts=False
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
                y_pred[left:right] == 1).any():
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


def soft_recall(y_true: np.ndarray,
                y_pred: np.ndarray,
                detection_range=3,
                return_counts=False,
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
        recall : float
            recall score
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
                y_pred[left:right] == 1).any():
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


def ctt(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Candidate To Target: Means distance between predicted anomaly and
    its closest true anomaly.

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels

    Returns
    -------
        ctt : float
            Candidate To Target time
    """
    if np.sum(y_true) == 0:
        # No anomalies to detect
        return float('inf')
    elif np.sum(y_pred) == 0:
        # No anomalies detected
        return 0

    tot_signed_dist = 0

    # Indices of true anomalies
    true_anomalies = np.where(y_true == 1)[0]

    for i in np.where(y_pred == 1)[0]:
        if len(true_anomalies) > 0:
            # signed distances to all true anomalies
            signed_dists = true_anomalies - i
            # signed distance with the smallest absolute value
            min_signed_dist = signed_dists[np.argmin(np.abs(signed_dists))]
            tot_signed_dist += min_signed_dist

    return tot_signed_dist / np.sum(y_pred)


def ttc(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Target To Candidate: Means distance between true anomaly and
    its closest predicted anomaly.

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels

    Returns
    -------
        ttc : float
            Target To Candidate
    """
    if np.sum(y_pred) == 0:
        # No true anomalies
        return float('inf')
    elif np.sum(y_true) == 0:
        # No anomalies detected
        return 0

    tot_signed_dist = 0

    # Indices of predicted anomalies
    pred_anomalies = np.where(y_pred == 1)[0]

    for i in np.where(y_true == 1)[0]:
        if len(pred_anomalies) > 0:
            # signed distances to all predicted anomalies
            signed_dists = pred_anomalies - i
            # signed distance with the smallest absolute value
            min_signed_dist = signed_dists[np.argmin(np.abs(signed_dists))]
            tot_signed_dist += min_signed_dist

    return tot_signed_dist / np.sum(y_true)
