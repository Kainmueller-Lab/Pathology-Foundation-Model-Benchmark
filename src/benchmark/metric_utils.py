import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def accuracy(y_true, y_pred, class_id=None):
    """
    Computes the accuracy score.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_id (int, optional): Class ID to compute accuracy for.

    Returns:
        float: Accuracy score.
    """
    if class_id is not None:
        y_pred = y_pred[y_true == class_id]
        y_true = y_true[y_true == class_id]
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, class_id):
    """
    Computes the precision for a specific class.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_id (int): Class ID to compute precision for.

    Returns:
        float: Precision score for the specified class.
    """
    return precision_score(
        y_true,
        y_pred,
        labels=[class_id] if class_id is not None else None,
        average='micro',
        zero_division=0,
    )

def recall(y_true, y_pred, class_id):
    """
    Computes the recall for a specific class.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_id (int): Class ID to compute recall for.

    Returns:
        float: Recall score for the specified class.
    """
    return recall_score(
        y_true,
        y_pred,
        labels=[class_id] if class_id is not None else None,
        average='micro',
        zero_division=0,
    )

def f1_score_class(y_true, y_pred, class_id):
    """
    Computes the F1 score for a specific class.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_id (int): Class ID to compute F1 score for.

    Returns:
        float: F1 score for the specified class.
    """
    return f1_score(
        y_true,
        y_pred,
        labels=[class_id] if class_id is not None else None,
        average='micro',
        zero_division=0,
    )

def confusion_matrix_func(y_true, y_pred, labels=None):
    """
    Computes the confusion matrix.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        labels (list, optional): List of labels to index the matrix.

    Returns:
        np.ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
