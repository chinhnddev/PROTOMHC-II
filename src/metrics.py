import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    idx = f1s.argmax()
    thr = thresholds[idx - 1] if idx > 0 else 0.5  # thresholds has len = len(p)-1
    return thr, f1s[idx]


def classification_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def ranking_metrics(y_true, y_prob, ks=(20, 50, 100)):
    order = np.argsort(-y_prob)
    y_sorted = np.array(y_true)[order]
    out = {}
    for k in ks:
        topk = y_sorted[:k] if len(y_sorted) >= k else y_sorted
        pos_in_data = sum(y_true)
        out[f"precision@{k}"] = topk.mean() if len(topk) else 0.0
        out[f"recall@{k}"] = topk.sum() / max(pos_in_data, 1)
    return out


def aggregate_metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob)
    thr, best_f1 = best_f1_threshold(y_true, y_prob)
    cls = classification_metrics(y_true, y_prob, thr)
    topk = ranking_metrics(y_true, y_prob)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": best_f1,
        "best_threshold": thr,
        **cls,
        **topk,
    }
