#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(name: str, command: list[str], skip: bool = False) -> None:
    if skip:
        print(f"\n=== SKIP {name} ===")
        return

    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the three CNN models: EfficientNet-B0, ConvNeXt-Tiny, DenseNet121."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for EfficientNet-B0 and ConvNeXt-Tiny.")
    parser.add_argument("--efficientnet-batch-size", type=int, default=16)
    parser.add_argument("--convnext-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-efficientnet", action="store_true")
    parser.add_argument("--skip-convnext", action="store_true")
    parser.add_argument("--skip-densenet", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    efficientnet_train = [
        PYTHON,
        "03-src/train_cnn.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.efficientnet_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--pretrained",
    ]
    efficientnet_eval = [
        PYTHON,
        "03-src/evaluate_cnn.py",
        "--checkpoint",
        "05-models/efficientnet_b0/best.pt",
    ]

    convnext_train = [
        PYTHON,
        "03-src/models/convnext_tiny/train_convnext_tiny.py",
        "--use-augmented-dir",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.convnext_batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    convnext_eval = [
        PYTHON,
        "03-src/models/convnext_tiny/evaluate_test.py",
    ]

    densenet_train = [
        PYTHON,
        "03-src/models/densenet121/train_densenet121.py",
    ]
    densenet_eval = [
        PYTHON,
        "03-src/models/densenet121/evaluate_test.py",
    ]

    run_step("TRAIN EfficientNet-B0", efficientnet_train, skip=args.skip_efficientnet)
    if not args.skip_evaluate:
        run_step("EVALUATE EfficientNet-B0", efficientnet_eval, skip=args.skip_efficientnet)

    run_step("TRAIN ConvNeXt-Tiny", convnext_train, skip=args.skip_convnext)
    if not args.skip_evaluate:
        run_step("EVALUATE ConvNeXt-Tiny", convnext_eval, skip=args.skip_convnext)

    run_step("TRAIN DenseNet121", densenet_train, skip=args.skip_densenet)
    if not args.skip_evaluate:
        run_step("EVALUATE DenseNet121", densenet_eval, skip=args.skip_densenet)

    print("\nDone.")


if __name__ == "__main__":
    main()
