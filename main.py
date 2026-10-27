#!/usr/bin/env python3
"""
Master Execution Entrypoint for Hyperspectral-Face-Identification-XAI
Provides a unified interface to execute Track 1 (RGB Baseline), Track 2 (Hyperspectral PUMA),
or display the Cross-Track Quantitative Benchmark.

Usage:
    python main.py                    # Displays cross-track benchmark matrix
    python main.py --track 2          # Runs Track 2 master research pipeline
    python main.py --track 2 --smoke  # Runs Track 2 15-second smoke test
    python main.py --track 1          # Runs Track 1 RGB baseline training
    python main.py --diagnose         # Runs hyperspectral dataset diagnostics
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Unified Entrypoint for Real-Time Hyperspectral Human Face Identification",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--track",
        type=int,
        choices=[1, 2],
        default=None,
        help="Specify which track to execute:\n"
             "  1: Track 1 (RGB Face Identification & XAI)\n"
             "  2: Track 2 (Hyperspectral Face Recognition with PUMA)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast verification smoke test (Track 2 only, ~15 seconds)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run zero-dependency dataset diagnostics on raw ENVI cubes",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Display cross-track quantitative comparison matrix",
    )

    args, extra_args = parser.parse_known_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Case 1: Dataset Diagnostics
    if args.diagnose:
        diag_script = os.path.join(root_dir, "track2_hyperspectral_puma", "diagnose_dataset.py")
        cmd = [sys.executable, diag_script, root_dir]
        print(f">>> Running Dataset Diagnostics: {' '.join(cmd)}\n")
        sys.exit(subprocess.call(cmd))

    # Case 2: Track 1 Execution
    if args.track == 1:
        t1_script = os.path.join(root_dir, "track1_rgb_baseline", "train_rgb_baseline.py")
        cmd = [sys.executable, t1_script] + extra_args
        print(f">>> Executing Track 1 (RGB Baseline): {' '.join(cmd)}\n")
        sys.exit(subprocess.call(cmd, cwd=os.path.join(root_dir, "track1_rgb_baseline")))

    # Case 3: Track 2 Execution
    if args.track == 2:
        if args.smoke:
            t2_script = os.path.join(root_dir, "track2_hyperspectral_puma", "scripts", "run_smoke_test.py")
        else:
            t2_script = os.path.join(root_dir, "track2_hyperspectral_puma", "main.py")
        cmd = [sys.executable, t2_script] + extra_args
        print(f">>> Executing Track 2 (Hyperspectral PUMA): {' '.join(cmd)}\n")
        sys.exit(subprocess.call(cmd, cwd=os.path.join(root_dir, "track2_hyperspectral_puma")))

    # Default / Benchmark Case
    from run_benchmark_comparison import main as show_benchmark
    show_benchmark()


if __name__ == "__main__":
    main()
