#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = PROJECT_ROOT / "04-features" / "handcrafted_features.npz"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "04-features" / "label_map.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "05-models" / "lightgbm_handcrafted"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


def require_lightgbm():
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install it with: pip install lightgbm"
        ) from exc
    return LGBMClassifier, early_stopping, log_evaluation


def load_label_map(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def clean_matrix(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    data = np.load(path, allow_pickle=True)
    required = ["X_train", "y_train", "X_val", "y_val"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing required arrays in {path}: {missing}")

    arrays = {
        "X_train": clean_matrix(data["X_train"]),
        "y_train": data["y_train"].astype(np.int64),
        "X_val": clean_matrix(data["X_val"]),
        "y_val": data["y_val"].astype(np.int64),
    }
    if "X_test" in data.files and "y_test" in data.files:
        arrays["X_test"] = clean_matrix(data["X_test"])
        arrays["y_test"] = data["y_test"].astype(np.int64)
    for key in ("train_paths", "val_paths", "test_paths"):
        if key in data.files:
            arrays[key] = data[key].astype(str)
    return arrays


def evaluate(model, X: np.ndarray, y: np.ndarray, target_names: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1)
    else:
        confidence = np.zeros(len(pred), dtype=np.float32)
    report = classification_report(
        y,
        pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    return metrics, pred.astype(np.int64), confidence.astype(np.float32)


def save_predictions(
    path: Path,
    sample_paths: np.ndarray | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    idx_to_class: dict[int, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "true_class", "pred_class", "confidence", "correct"])
        if sample_paths is None:
            sample_paths = np.asarray([""] * len(y_true), dtype=str)
        for image_path, true_label, pred_label, conf in zip(sample_paths, y_true, y_pred, confidence):
            writer.writerow(
                [
                    image_path,
                    int(true_label),
                    int(pred_label),
                    idx_to_class[int(true_label)],
                    idx_to_class[int(pred_label)],
                    float(conf),
                    int(true_label == pred_label),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM on handcrafted or hybrid feature matrices.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LGBMClassifier, early_stopping, log_evaluation = require_lightgbm()

    arrays = load_npz(args.features)
    label_map = load_label_map(args.label_map)
    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    num_classes = len(target_names)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=args.num_threads,
        verbosity=-1,
    )

    print(f"features: {args.features}")
    print(f"train: {arrays['X_train'].shape}, val: {arrays['X_val'].shape}")
    if "X_test" in arrays:
        print(f"test : {arrays['X_test'].shape}")

    model.fit(
        arrays["X_train"],
        arrays["y_train"],
        eval_set=[(arrays["X_val"], arrays["y_val"])],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(args.early_stopping_rounds, verbose=True),
            log_evaluation(period=25),
        ],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "lightgbm.joblib")

    params = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics = {
        "feature_path": str(args.features),
        "feature_dim": int(arrays["X_train"].shape[1]),
        "train_size": int(arrays["X_train"].shape[0]),
        "val_size": int(arrays["X_val"].shape[0]),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "params": params,
    }

    val_metrics, val_pred, val_conf = evaluate(model, arrays["X_val"], arrays["y_val"], target_names)
    metrics["val"] = val_metrics
    save_predictions(
        args.output_dir / "val_predictions.csv",
        arrays.get("val_paths"),
        arrays["y_val"],
        val_pred,
        val_conf,
        idx_to_class,
    )

    if "X_test" in arrays:
        test_metrics, test_pred, test_conf = evaluate(model, arrays["X_test"], arrays["y_test"], target_names)
        metrics["test"] = test_metrics
        metrics["test_size"] = int(arrays["X_test"].shape[0])
        save_predictions(
            args.output_dir / "test_predictions.csv",
            arrays.get("test_paths"),
            arrays["y_test"],
            test_pred,
            test_conf,
            idx_to_class,
        )

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults")
    for split in ("val", "test"):
        if split not in metrics:
            continue
        report = metrics[split]["classification_report"]
        print(
            f"{split:4s} acc={metrics[split]['accuracy']:.4f} "
            f"macro_f1={report['macro avg']['f1-score']:.4f} "
            f"weighted_f1={report['weighted avg']['f1-score']:.4f}"
        )
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
