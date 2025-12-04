"""Common metric helpers."""
from sklearn.metrics import roc_auc_score, average_precision_score


def binary_metrics(y_true, y_pred):
    return {
        "auroc": roc_auc_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_pred),
    }
