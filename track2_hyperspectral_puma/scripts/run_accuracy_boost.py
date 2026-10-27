"""
High-Accuracy Facial Recognition Evaluation Engine.
Quantifies the progression from isolated microscopic skin pixels to
spatial-spectral patch pooling and whole-face biometric consensus decisions.
"""

import csv
import json
import os
import sys
from typing import Dict, List, Any

# Ensure safe local matplotlib cache before any imports
os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.config import load_config
from src.preprocessing.snr_filter import SpectralBandFilter
from src.preprocessing.normalizer import SpectralNormalizer
from src.features.spatial_spectral import SpatialSpectralFeatureExtractor
from src.features.ensemble_classifier import HighCapacityEnsemble
from src.evaluation.face_level_recognition import WholeFaceDecisionEngine
from src.visualization.accuracy_plots import AccuracyVisualizer


def main():
    print("=" * 80)
    print("HIGH-ACCURACY FACE RECOGNITION BOOST ENGINE")
    print("=" * 80)

    cfg = load_config("configs/default_config.yaml")

    # 1. Load cached dataset
    cache_path = os.path.abspath("outputs/cache/face_dataset.npz")
    print(f"\n[1/5] Loading facial dataset from {cache_path}...")
    npz = np.load(cache_path, allow_pickle=True)
    X_train_raw = npz["X_train"]
    y_train = npz["y_train"]
    X_test_raw = npz["X_test"]
    y_test = npz["y_test"]
    raw_wavelengths = npz["wavelengths"]
    subject_names = [str(s) for s in npz["subject_names"]]

    # 2. Filter uncalibrated edges & apply L2 normalization
    print("[2/5] Filtering uncalibrated edges & applying L2 normalization...")
    band_filter = SpectralBandFilter(
        min_wavelength_nm=cfg.preprocessing.min_wavelength_nm,
        max_wavelength_nm=cfg.preprocessing.max_wavelength_nm,
    )
    band_filter.fit(raw_wavelengths, X_train_raw)
    valid_wavelengths = band_filter.valid_wavelengths

    X_train_filt = band_filter.transform(X_train_raw)
    X_test_filt = band_filter.transform(X_test_raw)

    normalizer = SpectralNormalizer(method="l2")
    X_tr_norm = normalizer.transform(X_train_filt)
    X_te_norm = normalizer.transform(X_test_filt)

    # 3. Load winning subsets
    winner_json_path = os.path.abspath("outputs/tables/three_band_winner.json")
    comp_json_path = os.path.abspath("outputs/tables/comparison_metrics.json")

    with open(winner_json_path, "r") as f:
        winner_data = json.load(f)
    best_3_bands = winner_data["best_three_bands_filtered"]

    with open(comp_json_path, "r") as f:
        comp_data = json.load(f)
    puma_30_bands = comp_data["PUMA"][0]["filtered_bands"]

    print(f"      PO-3 Band Indices : {best_3_bands} ({[round(float(valid_wavelengths[b]), 1) for b in best_3_bands]} nm)")
    print(f"      PUMA 30 Bands     : {len(puma_30_bands)} channels selected")

    rng = np.random.RandomState(42)
    sample_tr = min(len(X_tr_norm), 12000)
    sample_te = min(len(X_te_norm), 8000)
    idx_tr = rng.choice(len(X_tr_norm), size=sample_tr, replace=False)
    idx_te = rng.choice(len(X_te_norm), size=sample_te, replace=False)

    configurations = {
        "PO-3 Triad (3 Bands)": best_3_bands,
        "PUMA Best (30 Bands)": puma_30_bands,
        "Full Spectrum (925 Bands)": list(range(X_tr_norm.shape[1])),
    }

    progression_results = {name: {} for name in configurations}

    # 4. Multilevel Evaluation
    print("\n" + "=" * 80)
    print("[3/5] MULTILEVEL ACCURACY PROGRESSION EVALUATION")
    print("=" * 80)

    extractor_16 = SpatialSpectralFeatureExtractor(patch_size=16)
    extractor_64 = SpatialSpectralFeatureExtractor(patch_size=64)
    decision_engine = WholeFaceDecisionEngine(subject_names=subject_names)

    face_decisions_record = {}

    for name, b_idx in configurations.items():
        print(f"\n---> Evaluating Configuration: {name} ({len(b_idx)} bands)...")
        X_tr_sub = X_tr_norm[:, b_idx]
        X_te_sub = X_te_norm[:, b_idx]

        # Stage 1: Raw Isolated Pixel Level
        clf_pix = HighCapacityEnsemble(model_type="extra_trees", random_seed=42)
        clf_pix.fit(X_tr_sub[idx_tr], y_train[idx_tr])
        metrics_pix = clf_pix.evaluate(X_te_sub[idx_te], y_test[idx_te])
        acc_pix = metrics_pix["accuracy"]
        progression_results[name]["1. Isolated Pixel"] = acc_pix
        print(f"     [Stage 1] Raw Isolated Pixel Accuracy : {acc_pix * 100:.2f}%")

        # Stage 2: Micro-Patch (16-pixel neighborhood average)
        X_tr_p16, y_tr_p16 = extractor_16.aggregate_patches(X_tr_sub, y_train, patch_size=16)
        X_te_p16, y_te_p16 = extractor_16.aggregate_patches(X_te_sub, y_test, patch_size=16)

        idx_tr_p = rng.choice(len(X_tr_p16), size=min(len(X_tr_p16), 5000), replace=False)
        idx_te_p = rng.choice(len(X_te_p16), size=min(len(X_te_p16), 2500), replace=False)

        clf_p16 = HighCapacityEnsemble(model_type="extra_trees", random_seed=42)
        clf_p16.fit(X_tr_p16[idx_tr_p], y_tr_p16[idx_tr_p])
        acc_p16 = clf_p16.evaluate(X_te_p16[idx_te_p], y_te_p16[idx_te_p])["accuracy"]
        progression_results[name]["2. Micro-Patch (16px)"] = acc_p16
        print(f"     [Stage 2] Local Micro-Patch Accuracy  : {acc_p16 * 100:.2f}%")

        # Stage 3: Regional Block (64-pixel facial block)
        X_tr_p64, y_tr_p64 = extractor_64.aggregate_patches(X_tr_sub, y_train, patch_size=64)
        X_te_p64, y_te_p64 = extractor_64.aggregate_patches(X_te_sub, y_test, patch_size=64)

        clf_p64 = HighCapacityEnsemble(model_type="extra_trees", random_seed=42)
        clf_p64.fit(X_tr_p64, y_tr_p64)
        acc_p64 = clf_p64.evaluate(X_te_p64, y_te_p64)["accuracy"]
        progression_results[name]["3. Regional Block (64px)"] = acc_p64
        print(f"     [Stage 3] Regional Facial Block Acc   : {acc_p64 * 100:.2f}%")

        # Stage 4: Whole-Face ROI Decision Engine (Consensus Across Entire Face)
        face_res = decision_engine.evaluate_face_decisions(clf_p16, X_te_p16, y_te_p16)
        acc_face = face_res["whole_face_accuracy_hard"]
        progression_results[name]["4. Whole-Face ROI"] = acc_face
        face_decisions_record[name] = face_res
        print(f"     [Stage 4] WHOLE-FACE RECOGNITION ACC  : {acc_face * 100:.2f}% (Soft Margin: {face_res['whole_face_accuracy_soft']*100:.2f}%)")

    # 5. Export Tables
    print("\n[4/5] Exporting accuracy tables...")
    os.makedirs("outputs/tables", exist_ok=True)
    prog_csv = "outputs/tables/accuracy_progression_summary.csv"
    with open(prog_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Stage", "Accuracy_Percent"])
        for cfg_name, stages in progression_results.items():
            for stage_name, acc in stages.items():
                writer.writerow([cfg_name, stage_name, f"{acc*100:.2f}"])
    print(f"      Saved progression table -> {prog_csv}")

    face_csv = "outputs/tables/whole_face_recognition_decisions.csv"
    with open(face_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Subject", "True_ID", "Predicted_Hard", "Hard_Vote_Pct", "Correct_Hard", "Predicted_Soft", "Soft_Conf_Pct", "Confidence_Margin_Pct"])
        for cfg_name, res in face_decisions_record.items():
            for d in res["subject_decisions"]:
                writer.writerow([
                    cfg_name,
                    d["subject_name"],
                    d["subject_index"] + 1,
                    d["predicted_subject_hard"],
                    f"{d['hard_vote_pct']:.1f}",
                    d["is_correct_hard"],
                    d["predicted_subject_soft"],
                    f"{d['soft_confidence_pct']:.1f}",
                    f"{d['confidence_margin_pct']:.1f}",
                ])
    print(f"      Saved whole-face decisions -> {face_csv}")

    # 6. Generate Figures
    print("\n[5/5] Generating publication-grade figures...")
    os.makedirs("outputs/figures", exist_ok=True)
    vis = AccuracyVisualizer(output_dir="outputs/figures")

    p1 = vis.plot_accuracy_progression(progression_results, filename="accuracy_progression_bar.png")
    print(f"      Figure 1: Accuracy progression bar -> {p1}")

    cm_po3 = np.array(face_decisions_record["PO-3 Triad (3 Bands)"]["confusion_matrix_hard"])
    cm_puma30 = np.array(face_decisions_record["PUMA Best (30 Bands)"]["confusion_matrix_hard"])
    p2 = vis.plot_whole_face_confusion_matrices(cm_po3, cm_puma30, subject_names, filename="whole_face_confusion_matrix.png")
    print(f"      Figure 2: Whole-face confusion matrix -> {p2}")

    print("\n" + "=" * 80)
    print("HIGH-ACCURACY EVALUATION COMPLETE! ALL METRICS AND FIGURES READY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
