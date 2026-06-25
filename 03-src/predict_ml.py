#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib

from dataloader import IMAGE_EXTS, load_label_map
from feature_engineering import DEFAULT_LABEL_MAP, extract_handcrafted_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "05-models" / "handcrafted" / "random_forest.joblib"


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with handcrafted ML model.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")
    idx_to_class = {idx: cls for cls, idx in label_map.items()}

    paths = collect_images(args.input)
    if not paths:
        raise ValueError(f"No images found: {args.input}")

    rows = []
    for path in paths:
        features, _ = extract_handcrafted_features(path, image_size=args.image_size)
        pred = int(model.predict([features])[0])
        row = {
            "path": str(path),
            "pred_label": pred,
            "pred_class": idx_to_class[pred],
        }
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([features])[0]
            row["confidence"] = float(probs[pred])
        rows.append(row)

    for row in rows[:20]:
        confidence = row.get("confidence")
        suffix = f" ({confidence:.4f})" if confidence is not None else ""
        print(f"{row['path']} -> {row['pred_class']}{suffix}")
    if len(rows) > 20:
        print(f"... printed 20/{len(rows)} predictions")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
