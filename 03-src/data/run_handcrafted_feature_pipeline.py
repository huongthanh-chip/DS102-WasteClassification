#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "04-features"
DEFAULT_ANALYSIS_DIR = DEFAULT_FEATURE_DIR / "handcrafted_feature_analysis"
DEFAULT_LIGHTGBM_DIR = PROJECT_ROOT / "05-models" / "lightgbm_handcrafted_corr_filtered"


def run_step(name: str, command: list[str], skip: bool = False) -> None:
    if skip:
        print(f"\n=== SKIP {name} ===")
        return
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run handcrafted feature extraction, high-correlation filtering, "
            "and LightGBM training in one command."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--lightgbm-dir", type=Path, default=DEFAULT_LIGHTGBM_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--use-augmented-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-baseline-ml", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--skip-lightgbm", action="store_true")
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lightgbm-n-estimators", type=int, default=2000)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=63)
    parser.add_argument("--lightgbm-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--lightgbm-num-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_npz = args.feature_dir / "handcrafted_features.npz"
    feature_names = args.feature_dir / "handcrafted_feature_names.json"
    selected_npz = args.analysis_dir / "selected_handcrafted_features.npz"

    extract_cmd = [
        PYTHON,
        "03-src/data/feature_engineering.py",
        "--split-dir",
        str(args.split_dir),
        "--output-dir",
        str(args.feature_dir),
        "--image-size",
        str(args.image_size),
    ]
    if args.use_augmented_train:
        extract_cmd.append("--use-augmented-train")
    if args.train_baseline_ml:
        extract_cmd.append("--train-models")

    selection_cmd = [
        PYTHON,
        "03-src/data/analyze_handcrafted_features.py",
        "--features",
        str(feature_npz),
        "--feature-names",
        str(feature_names),
        "--output-dir",
        str(args.analysis_dir),
        "--correlation-threshold",
        str(args.correlation_threshold),
        "--variance-threshold",
        str(args.variance_threshold),
    ]

    lightgbm_cmd = [
        PYTHON,
        "03-src/models/ml/train_lightgbm.py",
        "--features",
        str(selected_npz),
        "--output-dir",
        str(args.lightgbm_dir),
        "--learning-rate",
        str(args.lightgbm_learning_rate),
        "--n-estimators",
        str(args.lightgbm_n_estimators),
        "--num-leaves",
        str(args.lightgbm_num_leaves),
        "--early-stopping-rounds",
        str(args.lightgbm_early_stopping_rounds),
        "--num-threads",
        str(args.lightgbm_num_threads),
    ]

    run_step("EXTRACT HANDCRAFTED FEATURES", extract_cmd, skip=args.skip_extract)
    run_step("FILTER HIGH-CORRELATION FEATURES", selection_cmd, skip=args.skip_selection)
    run_step("TRAIN LIGHTGBM", lightgbm_cmd, skip=args.skip_lightgbm)

    print("\nDone.")
    print(f"features        -> {feature_npz}")
    print(f"selected feature -> {selected_npz}")
    print(f"LightGBM output -> {args.lightgbm_dir}")


if __name__ == "__main__":
    main()
