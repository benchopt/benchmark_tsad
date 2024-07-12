import pytest  # noqa

import numpy as np
from benchmark_utils.metrics import soft_precision, soft_recall, ctt, ttc


def test_soft_precision():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert soft_precision(y1, y1, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p1 = y1.copy()
    assert soft_precision(y1, p1, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p2 = y1.copy()
    p2[7] = 0
    assert soft_precision(y1, p2, detection_range=1,
                          return_counts=True) == (1.0, 1, 0, 0)

    p3 = y1.copy()
    p3[8] = 1
    assert soft_precision(y1, p3, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p4 = y1.copy()
    p4[9] = 1
    assert soft_precision(y1, p4, detection_range=1,
                          return_counts=True) == (2/3, 2, 0, 1)

    p5 = np.zeros(10)
    p5[0] = 1
    assert soft_precision(y1, p5, detection_range=1,
                          return_counts=True) == (0.0, 0, 0, 1)

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert soft_precision(y1, p6, detection_range=1,
                          return_counts=True) == (1.0, 0, 2, 0)


def test_soft_recall():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert soft_recall(y1, y1, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p1 = y1.copy()
    assert soft_recall(y1, p1, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p2 = y1.copy()
    p2[7] = 0
    assert soft_recall(y1, p2, detection_range=1,
                       return_counts=True) == (0.5, 1, 0, 1)

    p3 = y1.copy()
    p3[8] = 1
    assert soft_recall(y1, p3, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p4 = y1.copy()
    p4[9] = 1
    assert soft_recall(y1, p4, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p5 = np.zeros(10)
    p5[0] = 1
    assert soft_recall(y1, p5, detection_range=1,
                       return_counts=True) == (0.0, 0, 0, 2)

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert soft_recall(y1, p6, detection_range=1,
                       return_counts=True) == (1.0, 0, 2, 0)


def test_ctt():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1

    p1 = y1.copy()
    assert ctt(y1, p1) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ctt(y1, p2) == 0.0

    p3 = y1.copy()
    p3[8] = 1
    assert ctt(y1, p3) == -1/3

    p4 = y1.copy()
    p4[9] = 1
    assert ctt(y1, p4) == -2/3

    p5 = np.zeros(10)
    p5[0] = 1
    assert ctt(y1, p5) == 3.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ctt(y1, p6) == -1.0


def test_ttc():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert ttc(y1, y1) == 0.0

    p1 = y1.copy()
    assert ttc(y1, p1) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ttc(y1, p2) == -2.0

    p3 = y1.copy()
    p3[8] = 1
    assert ttc(y1, p3) == 0.0

    p4 = y1.copy()
    p4[9] = 1
    assert ttc(y1, p4) == 0.0

    p5 = np.zeros(10)
    p5[0] = 1
    assert ttc(y1, p5) == -5.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ttc(y1, p6) == 1.0
