#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = PROJECT_ROOT / "04-features" / "handcrafted_features.npz"
DEFAULT_NAMES = PROJECT_ROOT / "04-features" / "handcrafted_feature_names.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "04-features" / "handcrafted_feature_analysis"


def load_feature_names(path: Path, n_features: int) -> list[str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)
        if len(names) == n_features:
            return [str(name) for name in names]
    return [f"feature_{idx:03d}" for idx in range(n_features)]


def finite_clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_correlation_pairs(
    X_scaled: np.ndarray,
    names: list[str],
    scores: np.ndarray,
    threshold: float,
    max_pairs: int,
) -> tuple[list[dict], set[int]]:
    corr = np.corrcoef(X_scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    rows: list[dict] = []
    drop: set[int] = set()
    n_features = corr.shape[0]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = abs(float(corr[i, j]))
            if value < threshold:
                continue
            if len(rows) < max_pairs:
                rows.append(
                    {
                        "feature_a": names[i],
                        "feature_b": names[j],
                        "abs_correlation": value,
                        "mi_a": float(scores[i]),
                        "mi_b": float(scores[j]),
                        "drop_candidate": names[i] if scores[i] < scores[j] else names[j],
                    }
                )
            if scores[i] < scores[j]:
                drop.add(i)
            else:
                drop.add(j)
    rows.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return rows, drop


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    variance_threshold: float,
    correlation_threshold: float,
    top_k: int | None,
    max_corr_pairs: int,
) -> tuple[list[int], list[dict], list[dict], dict]:
    X_train = finite_clean(X_train)
    variances = X_train.var(axis=0)
    non_low_variance = np.where(variances > variance_threshold)[0]
    if len(non_low_variance) == 0:
        raise ValueError("No features left after low-variance filtering.")

    X_var = X_train[:, non_low_variance]
    names_var = [feature_names[idx] for idx in non_low_variance]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)

    mi_scores = mutual_info_classif(X_scaled, y_train, discrete_features=False, random_state=42)
    corr_rows, corr_drop_local = build_correlation_pairs(
        X_scaled=X_scaled,
        names=names_var,
        scores=mi_scores,
        threshold=correlation_threshold,
        max_pairs=max_corr_pairs,
    )

    keep_local = [idx for idx in range(len(names_var)) if idx not in corr_drop_local]
    if top_k is not None and top_k > 0 and len(keep_local) > top_k:
        keep_local = sorted(keep_local, key=lambda idx: mi_scores[idx], reverse=True)[:top_k]
        keep_local = sorted(keep_local)

    selected_indices = [int(non_low_variance[idx]) for idx in keep_local]

    score_rows: list[dict] = []
    selected_set = set(selected_indices)
    dropped_corr_global = {int(non_low_variance[idx]) for idx in corr_drop_local}
    for local_idx, global_idx in enumerate(non_low_variance):
        reason = "selected"
        if global_idx not in selected_set:
            reason = "low_rank" if global_idx not in dropped_corr_global else "high_correlation"
        score_rows.append(
            {
                "feature_index": int(global_idx),
                "feature_name": feature_names[int(global_idx)],
                "variance": float(variances[int(global_idx)]),
                "mutual_info": float(mi_scores[local_idx]),
                "selected": int(global_idx in selected_set),
                "reason": reason,
            }
        )

    low_variance_rows = [
        {
            "feature_index": int(idx),
            "feature_name": feature_names[int(idx)],
            "variance": float(variances[int(idx)]),
        }
        for idx in np.where(variances <= variance_threshold)[0]
    ]

    summary = {
        "input_features": int(X_train.shape[1]),
        "low_variance_dropped": int(len(low_variance_rows)),
        "correlation_threshold": float(correlation_threshold),
        "high_correlation_dropped": int(len(dropped_corr_global)),
        "selected_features": int(len(selected_indices)),
        "top_k": top_k,
    }
    score_rows.sort(key=lambda row: row["mutual_info"], reverse=True)
    return selected_indices, score_rows, corr_rows, {"summary": summary, "low_variance": low_variance_rows}


def save_selected_npz(input_npz: Path, output_npz: Path, selected_indices: list[int]) -> None:
    data = np.load(input_npz, allow_pickle=True)
    arrays = {}
    for key in data.files:
        value = data[key]
        if key.startswith("X_") and value.ndim == 2:
            arrays[key] = finite_clean(value)[:, selected_indices]
        else:
            arrays[key] = value
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze handcrafted features and select a less redundant subset."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_NAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--max-corr-pairs", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)
    X_train = finite_clean(data["X_train"])
    y_train = data["y_train"].astype(np.int64)
    feature_names = load_feature_names(args.feature_names, X_train.shape[1])

    selected_indices, score_rows, corr_rows, extra = select_features(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold,
        top_k=args.top_k,
        max_corr_pairs=args.max_corr_pairs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = [feature_names[idx] for idx in selected_indices]
    selected_npz = args.output_dir / "selected_handcrafted_features.npz"

    save_selected_npz(args.features, selected_npz, selected_indices)
    np.save(args.output_dir / "selected_feature_indices.npy", np.asarray(selected_indices, dtype=np.int64))

    with open(args.output_dir / "selected_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(selected_names, f, indent=2)
    with open(args.output_dir / "feature_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(extra["summary"], f, indent=2)
    with open(args.output_dir / "low_variance_features.json", "w", encoding="utf-8") as f:
        json.dump(extra["low_variance"], f, indent=2)

    write_csv(
        args.output_dir / "feature_scores.csv",
        score_rows,
        ["feature_index", "feature_name", "variance", "mutual_info", "selected", "reason"],
    )
    write_csv(
        args.output_dir / "high_correlation_pairs.csv",
        corr_rows,
        ["feature_a", "feature_b", "abs_correlation", "mi_a", "mi_b", "drop_candidate"],
    )

    print(json.dumps(extra["summary"], indent=2))
    print(f"selected features -> {args.output_dir / 'selected_feature_names.json'}")
    print(f"selected matrix   -> {selected_npz}")


if __name__ == "__main__":
    main()
