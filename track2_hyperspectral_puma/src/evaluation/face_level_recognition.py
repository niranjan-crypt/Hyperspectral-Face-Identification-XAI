"""
Whole-Face ROI Decision Engine.
Implements biometric consensus aggregation across facial bounding boxes,
converting microscopic pixel-level predictions into robust whole-face identifications.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class WholeFaceDecisionEngine:
    """Evaluates face-level biometric identification decisions across subjects."""

    def __init__(self, subject_names: List[str]):
        self.subject_names = subject_names
        self.n_classes = len(subject_names)

    def evaluate_face_decisions(
        self,
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Groups test predictions by true subject identity and computes
        both hard majority voting and soft probability pooling decisions.
        """
        has_proba = hasattr(classifier, "predict_proba")
        if has_proba:
            all_probas = classifier.predict_proba(X_test)
        else:
            all_probas = None

        all_preds = classifier.predict(X_test)

        subject_decisions = []
        true_labels = []
        predicted_labels_hard = []
        predicted_labels_soft = []

        for c in range(self.n_classes):
            mask = (y_test == c)
            if not np.any(mask):
                continue

            true_labels.append(c)
            sub_preds = all_preds[mask]

            # 1. Hard Majority Vote
            counts = np.bincount(sub_preds, minlength=self.n_classes)
            pred_hard = int(np.argmax(counts))
            vote_pct = float(counts[pred_hard] / len(sub_preds) * 100)
            predicted_labels_hard.append(pred_hard)

            # 2. Soft Probability Pooling (if model supports it)
            if has_proba:
                sub_probas = all_probas[mask]
                mean_probas = np.mean(sub_probas, axis=0)
                pred_soft = int(np.argmax(mean_probas))
                soft_confidence = float(mean_probas[pred_soft] * 100)
                # Compute margin over runner-up
                sorted_p = np.sort(mean_probas)[::-1]
                margin = float((sorted_p[0] - sorted_p[1]) * 100) if len(sorted_p) > 1 else 100.0
            else:
                pred_soft = pred_hard
                soft_confidence = vote_pct
                margin = 0.0

            predicted_labels_soft.append(pred_soft)

            decision_record = {
                "subject_index": c,
                "subject_name": self.subject_names[c],
                "pixels_evaluated": int(np.sum(mask)),
                "predicted_subject_hard": self.subject_names[pred_hard],
                "hard_vote_pct": vote_pct,
                "is_correct_hard": bool(pred_hard == c),
                "predicted_subject_soft": self.subject_names[pred_soft],
                "soft_confidence_pct": soft_confidence,
                "confidence_margin_pct": margin,
                "is_correct_soft": bool(pred_soft == c),
            }
            subject_decisions.append(decision_record)

        # Whole-face accuracy
        oa_hard = float(np.mean([d["is_correct_hard"] for d in subject_decisions]))
        oa_soft = float(np.mean([d["is_correct_soft"] for d in subject_decisions]))

        # Confusion matrix
        cm_hard = confusion_matrix(true_labels, predicted_labels_hard, labels=list(range(self.n_classes)))
        cm_soft = confusion_matrix(true_labels, predicted_labels_soft, labels=list(range(self.n_classes)))

        return {
            "whole_face_accuracy_hard": oa_hard,
            "whole_face_accuracy_soft": oa_soft,
            "subject_decisions": subject_decisions,
            "confusion_matrix_hard": cm_hard.tolist(),
            "confusion_matrix_soft": cm_soft.tolist(),
        }
