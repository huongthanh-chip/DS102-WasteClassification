#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def infer_true_label(path: Path, label_map: dict[str, int]) -> tuple[int | None, str | None]:
    for parent in [path.parent, *path.parents]:
        if parent.name in label_map:
            return label_map[parent.name], parent.name
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with handcrafted ML model.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "n_jobs"):
                step.n_jobs = 1

    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")
    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    paths = collect_images(args.input)
    if not paths:
        raise ValueError(f"No images found: {args.input}")

    rows = []
    y_true = []
    y_pred = []
    for path in paths:
        features, _ = extract_handcrafted_features(path, image_size=args.image_size)
        pred = int(model.predict([features])[0])
        true_label, true_class = infer_true_label(path, label_map)
        row = {
            "path": str(path),
            "true_label": true_label,
            "true_class": true_class,
            "pred_label": pred,
            "pred_class": idx_to_class[pred],
        }
        if true_label is not None:
            row["correct"] = int(true_label == pred)
            y_true.append(true_label)
            y_pred.append(pred)
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

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            labels=sorted(idx_to_class),
            target_names=target_names,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_class)).tolist()
        print("\n=== Metrics ===")
        print(f"accuracy: {acc:.4f}")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=sorted(idx_to_class),
                target_names=target_names,
                digits=4,
                zero_division=0,
            )
        )
        print("confusion_matrix:")
        print(cm)

        metrics_path = args.metrics_json
        if metrics_path is None and args.output_csv is not None:
            metrics_path = args.output_csv.with_suffix(".metrics.json")
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input": str(args.input),
                        "model": str(args.model),
                        "accuracy": acc,
                        "classification_report": report,
                        "confusion_matrix": cm,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"metrics saved -> {metrics_path}")
    else:
        print("\n[INFO] No ground-truth labels inferred from folder names; metrics skipped.")

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
