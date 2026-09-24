"""
Evaluation metrics for face identification and biometric security:
- Classification accuracy, precision, recall, F1-score (Eq. 3.63 - 3.66)
- False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER) (Eq. 3.67 - 3.69)
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_classification_metrics(y_true, y_pred, average="weighted"):
    """
    Computes standard classification metrics (Eq. 3.63 - 3.66).
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }


def compute_biometric_rates(y_true_binary, scores, threshold=0.50):
    """
    Computes FAR and FRR for verification (Eq. 3.67 - 3.69).
    y_true_binary: 1 for genuine pair, 0 for imposter pair.
    """
    y_pred_binary = (np.array(scores) >= threshold).astype(int)

    # Imposter matches accepted (False Positives)
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    far = fp / (fp + tn + 1e-7)  # Eq. 3.67

    # Genuine matches rejected (False Negatives)
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    frr = fn / (fn + tp + 1e-7)  # Eq. 3.68

    return {"FAR": float(far), "FRR": float(frr)}


def compute_eer(y_true_binary, scores, num_thresholds=200):
    """
    Finds threshold where FAR == FRR (Equal Error Rate, Eq. 3.69).
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fars, frrs = [], []

    for th in thresholds:
        rates = compute_biometric_rates(y_true_binary, scores, threshold=th)
        fars.append(rates["FAR"])
        frrs.append(rates["FRR"])

    diff = np.abs(np.array(fars) - np.array(frrs))
    best_idx = int(np.argmin(diff))
    eer = (fars[best_idx] + frrs[best_idx]) / 2.0
    return float(eer), float(thresholds[best_idx])
