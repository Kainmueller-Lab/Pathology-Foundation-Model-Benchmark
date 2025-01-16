import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from benchmark.metric_utils import (
    accuracy,
    precision,
    recall,
    f1_score_class,
    confusion_matrix_func,
)


def test_accuracy():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert accuracy(y_true, y_pred) == accuracy_score(y_true, y_pred)
    assert accuracy(y_true, y_pred, class_id=1) == accuracy_score(
        y_true[y_true == 1], y_pred[y_true == 1]
    )


def test_precision():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert precision(y_true, y_pred, class_id=1) == precision_score(
        y_true, y_pred, labels=[1], average="micro", zero_division=0
    )


def test_recall():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert recall(y_true, y_pred, class_id=1) == recall_score(
        y_true, y_pred, labels=[1], average="micro", zero_division=0
    )


def test_f1_score_class():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert f1_score_class(y_true, y_pred, class_id=1) == f1_score(
        y_true, y_pred, labels=[1], average="micro", zero_division=0
    )


def test_confusion_matrix_func():
    y_true = np.array([0, 1, 2, 2, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert np.array_equal(confusion_matrix_func(y_true, y_pred), confusion_matrix(y_true, y_pred))
    assert np.array_equal(
        confusion_matrix_func(y_true, y_pred, labels=[0, 1, 2]),
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    )
