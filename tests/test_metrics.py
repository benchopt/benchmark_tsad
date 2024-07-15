import pytest  # noqa

import numpy as np
from benchmark_utils.metrics import (
    soft_precision, soft_recall, soft_f1, ctt, ttc
)


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


def test_soft_f1():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1

    precision = soft_precision(y1, y1, detection_range=1)
    recall = soft_recall(y1, y1, detection_range=1)

    assert soft_f1(y1, y1, detection_range=1) == 1.0

    p2 = y1.copy()
    p2[7] = 0
    precision = soft_precision(y1, p2, detection_range=1)
    recall = soft_recall(y1, p2, detection_range=1)

    assert soft_f1(y1, p2, detection_range=1) == 2 * \
        precision * recall / (precision + recall)

    p3 = y1.copy()
    p3[8] = 1

    precision = soft_precision(y1, p3, detection_range=1)
    recall = soft_recall(y1, p3, detection_range=1)

    assert soft_f1(y1, p3, detection_range=1) == 2 * \
        precision * recall / (precision + recall)


def test_ctt():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1

    p1 = y1.copy()
    assert ctt(y1, p1, return_signed=True) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ctt(y1, p2, return_signed=True) == 0.0

    p3 = y1.copy()
    p3[8] = 1
    assert ctt(y1, p3, return_signed=True) == -1/3

    p4 = y1.copy()
    p4[9] = 1
    assert ctt(y1, p4, return_signed=True) == -2/3

    p5 = np.zeros(10)
    p5[0] = 1
    assert ctt(y1, p5, return_signed=True) == 3.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ctt(y1, p6, return_signed=True) == -1.0

    y2 = np.zeros(10)
    p7 = np.zeros(10)
    p7[4] = p7[8] = 1
    assert ctt(y2, p7, return_signed=True) == float('inf')

    p8 = np.zeros(10)
    assert ctt(y2, p8, return_signed=True) == float('inf')

    y3 = np.zeros(10)
    y3[2] = 1
    p9 = np.zeros(10)
    assert ctt(y3, p9, return_signed=True) == 0.0


def test_ttc():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert ttc(y1, y1, return_signed=True) == 0.0

    p1 = y1.copy()
    assert ttc(y1, p1, return_signed=True) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ttc(y1, p2, return_signed=True) == -2.0

    p3 = y1.copy()
    p3[8] = 1
    assert ttc(y1, p3, return_signed=True) == 0.0

    p4 = y1.copy()
    p4[9] = 1
    assert ttc(y1, p4, return_signed=True) == 0.0

    p5 = np.zeros(10)
    p5[0] = 1
    assert ttc(y1, p5, return_signed=True) == -5.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ttc(y1, p6, return_signed=True) == 1.0

    y2 = np.zeros(10)
    p7 = np.zeros(10)
    p7[4] = p7[8] = 1
    assert ttc(y2, p7, return_signed=True) == 0

    p8 = np.zeros(10)
    assert ttc(y2, p8, return_signed=True) == float('inf')

    y3 = np.zeros(10)
    y3[2] = 1
    p9 = np.zeros(10)
    assert ttc(y3, p9, return_signed=True) == float('inf')
