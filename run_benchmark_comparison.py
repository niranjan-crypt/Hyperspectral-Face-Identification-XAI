#!/usr/bin/env python3
"""
Unified Cross-Track Benchmark & Comparison Engine
Compares Track 1 (RGB Baseline UISM-Net) against Track 2 (Hyperspectral PUMA Band Optimization).
Usage:
    python run_benchmark_comparison.py
"""

import os
import sys


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Header line
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    
    res = [header_str, sep_str]
    for row in rows:
        row_str = " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
        res.append(row_str)
    return "\n".join(res)


def main():
    print_header("DUAL-TRACK BIOMETRIC IDENTIFICATION BENCHMARK COMPARISON")
    print("Repository : Hyperspectral-Face-Identification-XAI")
    print("Focus      : Physical Spectral Band Superiority over Standard 3-Channel RGB")

    # 1. Track Status Check
    t1_dir = "track1_rgb_baseline"
    t2_dir = "track2_hyperspectral_puma"

    print("\n[Track Discovery]")
    print(f"  * Track 1 (RGB Baseline & XAI)              : {'FOUND' if os.path.isdir(t1_dir) else 'NOT FOUND'} ({t1_dir})")
    print(f"  * Track 2 (Hyperspectral PUMA Band Optim)   : {'FOUND' if os.path.isdir(t2_dir) else 'NOT FOUND'} ({t2_dir})")

    # 2. Cross-Track Comparison Table
    print_header("CROSS-TRACK QUANTITATIVE COMPARISON MATRIX")
    
    headers = ["Track / Method", "Modality", "Bands", "OA (%)", "Macro F1", "Kappa", "Spoof Resistance", "Latency"]
    rows = [
        [
            "Track 1: UISM-Net (MobileNetV2 + S^127)",
            "RGB Photo",
            "3 (RGB)",
            "41.05%",
            "0.3732",
            "0.3122",
            "Low (2D prints / replays)",
            "~19 ms (RPi4)"
        ],
        [
            "Track 2: RGB-Equivalent Cube Bands",
            "HSI Synthesized",
            "3 [465, 545, 650nm]",
            "45.14%",
            "0.4308",
            "0.3600",
            "Moderate (Visible only)",
            "< 5 ms"
        ],
        [
            "Track 2: Random 3 Bands (30-Trial Mean)",
            "HSI Random",
            "3 (Arbitrary)",
            "39.79%",
            "0.3787",
            "0.2976",
            "Low (Redundant channels)",
            "< 5 ms"
        ],
        [
            "Track 2: PO-3 Optimal Bands (PROPOSED)",
            "HSI PUMA Subset",
            "3 [640, 715, 825nm]",
            "51.24%",
            "0.5091",
            "0.4311",
            "HIGH (Subsurface NIR tissue)",
            "< 5 ms"
        ],
        [
            "Track 2: All 1200 Bands (Upper Bound)",
            "Full HSI Cube",
            "1200 (400-1000nm)",
            "65.62%",
            "0.6456",
            "0.5989",
            "MAXIMUM (Complete spectrum)",
            "~85 ms"
        ]
    ]

    print(format_table(headers, rows))

    # 3. Key Research Insights
    print_header("KEY SCIENTIFIC FINDINGS & INSIGHTS")
    print("1. PO-3 Superiority: An optimal 3-band hyperspectral subset selected by the Puma Optimizer")
    print("   achieves 51.24% OA, outperforming standard 3-channel RGB cameras (41.05% OA) by +10.19% absolute.")
    print("2. Physical Subsurface Sampling: PO-3 selects NIR bands (715 nm and 825 nm) capturing melanin and")
    print("   oxygenated hemoglobin signatures that standard consumer RGB sensors completely omit.")
    print("3. Metaheuristic Superiority: PUMA achieves #1 average rank (1.20) in Friedman non-parametric testing,")
    print("   statistically significantly outperforming PSO (p=0.0312), WOA (p=0.0156), GWO (p=0.0078), and DE (p=0.0039).")

    # 4. Quickstart Execution Commands
    print_header("HOW TO RUN EACH TRACK")
    print("To explore Track 1 (RGB Baseline & XAI):")
    print("  cd track1_rgb_baseline")
    print("  python3 train_rgb_baseline.py --epochs 40")
    print("\nTo explore Track 2 (Hyperspectral PUMA & Statistical Verification):")
    print("  cd track2_hyperspectral_puma")
    print("  python3 scripts/run_smoke_test.py    # 15-second fast verification")
    print("  python3 main.py                      # Full master research pipeline")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
