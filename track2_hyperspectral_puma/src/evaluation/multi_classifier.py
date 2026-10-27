"""
Multi-Classifier Generalization Suite.
Evaluates selected band subsets across 4 distinct classifier paradigms:
    1. Nearest Centroid (Angle-based Cosine Prototype)
    2. Support Vector Machine (Non-linear RBF Kernel)
    3. Random Forest (Non-linear Bagging Ensemble)
    4. k-Nearest Neighbors (Distance-Weighted Non-parametric)
Proves that PUMA's spectral representations generalize effectively across diverse models.
"""

from typing import Dict, List, Sequence, Any
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from ..features.classifier import FastNearestCentroid


class MultiClassifierBenchmarker:
    """Evaluates band subsets across diverse classification architectures."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def evaluate_all_classifiers(
        self,
        X_train_sub: np.ndarray,
        y_train: np.ndarray,
        X_test_sub: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Trains and tests 4 classifier paradigms on the exact same band subset."""
        results = {}

        n_train = min(len(X_train_sub), 10000)
        n_test = min(len(X_test_sub), 8000)
        rng = np.random.RandomState(self.seed)

        idx_tr = rng.choice(len(X_train_sub), size=n_train, replace=False)
        idx_te = rng.choice(len(X_test_sub), size=n_test, replace=False)

        X_tr = X_train_sub[idx_tr]
        y_tr = y_train[idx_tr]
        X_te = X_test_sub[idx_te]
        y_te = y_test[idx_te]

        # 1. Cosine Nearest Centroid
        nc = FastNearestCentroid(metric="cosine").fit(X_tr, y_tr)
        preds_nc = nc.predict(X_te)
        results["Nearest_Centroid"] = {
            "accuracy": float(accuracy_score(y_te, preds_nc)),
            "f1_macro": float(f1_score(y_te, preds_nc, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_te, preds_nc)),
        }

        # 2. Non-linear RBF Support Vector Machine (scaled)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:5000])
        y_tr_s = y_tr[:5000]
        X_te_s = scaler.transform(X_te)
        svm = SVC(C=10.0, kernel="rbf", gamma="scale", random_state=self.seed)
        svm.fit(X_tr_s, y_tr_s)
        preds_svm = svm.predict(X_te_s)
        results["SVM_RBF"] = {
            "accuracy": float(accuracy_score(y_te, preds_svm)),
            "f1_macro": float(f1_score(y_te, preds_svm, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_te, preds_svm)),
        }

        # 3. Random Forest (150 Trees, tuned depth)
        rf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=3, random_state=self.seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        preds_rf = rf.predict(X_te)
        results["Random_Forest"] = {
            "accuracy": float(accuracy_score(y_te, preds_rf)),
            "f1_macro": float(f1_score(y_te, preds_rf, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_te, preds_rf)),
        }

        # 4. Distance-Weighted k-Nearest Neighbors
        knn = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine", n_jobs=-1)
        knn.fit(X_tr, y_tr)
        preds_knn = knn.predict(X_te)
        results["kNN"] = {
            "accuracy": float(accuracy_score(y_te, preds_knn)),
            "f1_macro": float(f1_score(y_te, preds_knn, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_te, preds_knn)),
        }

        return results
