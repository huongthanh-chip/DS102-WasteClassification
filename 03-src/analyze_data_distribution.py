#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "04-features" / "data_distribution_analysis"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "04-features" / "label_map.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RANDOM_STATE = 42


def load_label_map(path: str | Path = DEFAULT_LABEL_MAP) -> dict[str, int] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {str(k): int(v) for k, v in label_map.items()}


def list_images(split_dir: Path, split: str, label_map: dict[str, int]) -> list[tuple[Path, int, str]]:
    rows = []
    split_root = split_dir / split
    nested_split_root = split_root / split
    if nested_split_root.exists():
        split_root = nested_split_root

    for class_name, label in sorted(label_map.items(), key=lambda item: item[1]):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((path, label, class_name))
    return rows


def sample_rows(
    rows: list[tuple[Path, int, str]],
    max_per_class: int | None,
    seed: int,
) -> list[tuple[Path, int, str]]:
    if max_per_class is None:
        return rows

    rng = random.Random(seed)
    grouped: dict[str, list[tuple[Path, int, str]]] = {}
    for row in rows:
        grouped.setdefault(row[2], []).append(row)

    sampled = []
    for class_name in sorted(grouped):
        class_rows = grouped[class_name]
        if len(class_rows) > max_per_class:
            sampled.extend(rng.sample(class_rows, max_per_class))
        else:
            sampled.extend(class_rows)
    return sorted(sampled, key=lambda item: str(item[0]))


def image_stats(path: Path, image_size: int) -> dict[str, float]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    image = image.resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0

    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    brightness = arr.mean(axis=2)
    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)
    saturation = np.where(max_rgb > 1e-7, (max_rgb - min_rgb) / (max_rgb + 1e-7), 0.0)

    gray = brightness
    gy, gx = np.gradient(gray)
    edge_strength = np.sqrt(gx * gx + gy * gy)

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / max(height, 1)),
        "r_mean": float(rgb_mean[0]),
        "g_mean": float(rgb_mean[1]),
        "b_mean": float(rgb_mean[2]),
        "r_std": float(rgb_std[0]),
        "g_std": float(rgb_std[1]),
        "b_std": float(rgb_std[2]),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "contrast": float(gray.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "edge_strength_mean": float(edge_strength.mean()),
        "edge_strength_std": float(edge_strength.std()),
    }


def summarize_numeric(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    summary = {"n": float(len(rows))}
    if not rows:
        for key in keys:
            summary[f"{key}_mean"] = 0.0
            summary[f"{key}_std"] = 0.0
        return summary

    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_distribution_rows(
    split_rows: dict[str, list[tuple[Path, int, str]]],
    label_map: dict[str, int],
) -> list[dict]:
    rows = []
    for split, items in split_rows.items():
        total = len(items)
        counts = Counter(class_name for _, _, class_name in items)
        for class_name, label in sorted(label_map.items(), key=lambda item: item[1]):
            count = counts.get(class_name, 0)
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "label": label,
                    "count": count,
                    "percent": count / max(total, 1),
                }
            )
    return rows


def split_summary_rows(split_rows: dict[str, list[tuple[Path, int, str]]]) -> list[dict]:
    total = sum(len(rows) for rows in split_rows.values())
    return [
        {"split": split, "count": len(rows), "percent_of_total": len(rows) / max(total, 1)}
        for split, rows in split_rows.items()
    ]


def build_image_statistics(
    split_rows: dict[str, list[tuple[Path, int, str]]],
    image_size: int,
    max_per_class: int | None,
    seed: int,
) -> tuple[list[dict], list[dict], list[str]]:
    stat_keys = [
        "width",
        "height",
        "aspect_ratio",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "brightness_mean",
        "brightness_std",
        "contrast",
        "saturation_mean",
        "saturation_std",
        "edge_strength_mean",
        "edge_strength_std",
    ]
    per_image_rows = []
    aggregate_rows = []

    for split, rows in split_rows.items():
        sampled_rows = sample_rows(rows, max_per_class=max_per_class, seed=seed)
        by_class: dict[str, list[dict[str, float]]] = {}
        split_stats = []

        for idx, (path, label, class_name) in enumerate(sampled_rows, start=1):
            try:
                stats = image_stats(path, image_size=image_size)
            except Exception as exc:
                print(f"[WARN] skipped {path}: {exc}")
                continue
            record = {
                "split": split,
                "class_name": class_name,
                "label": label,
                "path": str(path),
                **stats,
            }
            per_image_rows.append(record)
            by_class.setdefault(class_name, []).append(stats)
            split_stats.append(stats)
            if idx % 500 == 0:
                print(f"{split}: analyzed {idx}/{len(sampled_rows)} images")

        for class_name in sorted(by_class):
            summary = summarize_numeric(by_class[class_name], stat_keys)
            aggregate_rows.append({"split": split, "class_name": class_name, **summary})

        summary = summarize_numeric(split_stats, stat_keys)
        aggregate_rows.append({"split": split, "class_name": "__all__", **summary})

    return per_image_rows, aggregate_rows, stat_keys


def lookup_label_percent(rows: list[dict]) -> dict[tuple[str, str], float]:
    return {(row["split"], row["class_name"]): float(row["percent"]) for row in rows}


def lookup_aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["split"], row["class_name"]): row for row in rows}


def distribution_drift_rows(
    label_rows: list[dict],
    aggregate_rows: list[dict],
    label_map: dict[str, int],
    stat_keys: list[str],
    reference_split: str = "train",
) -> list[dict]:
    label_pct = lookup_label_percent(label_rows)
    aggregates = lookup_aggregate(aggregate_rows)
    rows = []

    for split in sorted({row["split"] for row in label_rows}):
        if split == reference_split:
            continue
        for class_name in [*label_map.keys(), "__all__"]:
            record = {
                "reference_split": reference_split,
                "compared_split": split,
                "class_name": class_name,
            }
            if class_name != "__all__":
                record["label_percent_abs_diff"] = abs(
                    label_pct.get((split, class_name), 0.0)
                    - label_pct.get((reference_split, class_name), 0.0)
                )
            else:
                record["label_percent_abs_diff"] = 0.0

            ref = aggregates.get((reference_split, class_name))
            cmp = aggregates.get((split, class_name))
            max_standardized_diff = 0.0
            mean_abs_diff = 0.0
            n_stats = 0
            if ref and cmp:
                for key in stat_keys:
                    ref_mean = float(ref[f"{key}_mean"])
                    cmp_mean = float(cmp[f"{key}_mean"])
                    ref_std = float(ref[f"{key}_std"])
                    cmp_std = float(cmp[f"{key}_std"])
                    pooled = math.sqrt((ref_std * ref_std + cmp_std * cmp_std) / 2.0)
                    abs_diff = abs(cmp_mean - ref_mean)
                    std_diff = abs_diff / max(pooled, 1e-7)
                    max_standardized_diff = max(max_standardized_diff, std_diff)
                    mean_abs_diff += abs_diff
                    n_stats += 1
            record["mean_numeric_abs_diff"] = mean_abs_diff / max(n_stats, 1)
            record["max_numeric_standardized_diff"] = max_standardized_diff
            rows.append(record)
    return rows


def make_report(
    split_rows: dict[str, list[tuple[Path, int, str]]],
    drift_rows: list[dict],
    label_threshold: float,
    numeric_threshold: float,
) -> dict:
    max_label_diff = max((float(row["label_percent_abs_diff"]) for row in drift_rows), default=0.0)
    max_numeric_diff = max(
        (float(row["max_numeric_standardized_diff"]) for row in drift_rows),
        default=0.0,
    )
    return {
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "max_label_percent_abs_diff": max_label_diff,
        "max_numeric_standardized_diff": max_numeric_diff,
        "label_threshold": label_threshold,
        "numeric_threshold": numeric_threshold,
        "label_distribution_ok": max_label_diff <= label_threshold,
        "numeric_distribution_ok": max_numeric_diff <= numeric_threshold,
        "note": (
            "Use this as a diagnostic, not a statistical proof. Small classes can show "
            "larger per-class numeric drift even when stratified splitting is correct."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze train/val/test distribution drift.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--include-augmented-train", action="store_true")
    parser.add_argument("--label-threshold", type=float, default=0.02)
    parser.add_argument("--numeric-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")

    split_names = ["train", "val", "test"]
    if args.include_augmented_train:
        split_names.append("train_augmented")

    split_rows = {
        split: list_images(args.split_dir, split, label_map)
        for split in split_names
    }
    split_rows = {split: rows for split, rows in split_rows.items() if rows}

    print({split: len(rows) for split, rows in split_rows.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_rows = label_distribution_rows(split_rows, label_map)
    write_csv(
        args.output_dir / "class_distribution.csv",
        label_rows,
        ["split", "class_name", "label", "count", "percent"],
    )

    summary_rows = split_summary_rows(split_rows)
    write_csv(
        args.output_dir / "split_summary.csv",
        summary_rows,
        ["split", "count", "percent_of_total"],
    )

    per_image_rows, aggregate_rows, stat_keys = build_image_statistics(
        split_rows=split_rows,
        image_size=args.image_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    per_image_fields = ["split", "class_name", "label", "path", *stat_keys]
    write_csv(args.output_dir / "image_statistics.csv", per_image_rows, per_image_fields)

    aggregate_fields = ["split", "class_name", "n"]
    for key in stat_keys:
        aggregate_fields.extend([f"{key}_mean", f"{key}_std"])
    write_csv(args.output_dir / "aggregate_image_statistics.csv", aggregate_rows, aggregate_fields)

    drift_rows = distribution_drift_rows(label_rows, aggregate_rows, label_map, stat_keys)
    write_csv(
        args.output_dir / "distribution_drift_vs_train.csv",
        drift_rows,
        [
            "reference_split",
            "compared_split",
            "class_name",
            "label_percent_abs_diff",
            "mean_numeric_abs_diff",
            "max_numeric_standardized_diff",
        ],
    )

    report = make_report(
        split_rows=split_rows,
        drift_rows=drift_rows,
        label_threshold=args.label_threshold,
        numeric_threshold=args.numeric_threshold,
    )
    with open(args.output_dir / "distribution_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"saved -> {args.output_dir}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
