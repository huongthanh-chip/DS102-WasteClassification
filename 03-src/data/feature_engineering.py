#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from dataloader import IMAGE_EXTS, load_label_map


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "04-features"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "05-models"
DEFAULT_LABEL_MAP = DEFAULT_FEATURE_DIR / "label_map.json"


def list_images(split_dir: Path, split: str, label_map: dict[str, int]):
    rows = []
    split_root = split_dir / split
    for class_name, label in sorted(label_map.items(), key=lambda item: item[1]):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((path, label, class_name))
    return rows


def _channel_stats(arr: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    feats: list[float] = []
    names: list[str] = []
    for idx, channel_name in enumerate(("0", "1", "2")):
        channel = arr[:, :, idx].astype(np.float32)
        feats.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(np.percentile(channel, 25)),
                float(np.percentile(channel, 50)),
                float(np.percentile(channel, 75)),
            ]
        )
        names.extend(
            [
                f"{prefix}_{channel_name}_mean",
                f"{prefix}_{channel_name}_std",
                f"{prefix}_{channel_name}_p25",
                f"{prefix}_{channel_name}_p50",
                f"{prefix}_{channel_name}_p75",
            ]
        )
    return feats, names


def _safe_skew_kurt(channel: np.ndarray) -> tuple[float, float]:
    flat = channel.astype(np.float32).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    if std < 1e-8:
        return 0.0, 0.0
    z = (flat - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return skew, kurt


def _color_moments(arr: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    feats: list[float] = []
    names: list[str] = []
    for idx, channel_name in enumerate(("0", "1", "2")):
        skew, kurt = _safe_skew_kurt(arr[:, :, idx])
        feats.extend([skew, kurt])
        names.extend([f"{prefix}_{channel_name}_skew", f"{prefix}_{channel_name}_kurtosis"])
    return feats, names


def _entropy_from_uint8(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float32)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    entropy = float(-(prob * np.log2(prob)).sum())
    return [entropy], [f"{prefix}_entropy"]


def _ratio_features(rgb: np.ndarray, gray: np.ndarray, hsv: np.ndarray) -> tuple[list[float], list[str]]:
    gray_norm = gray.astype(np.float32) / 255.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    white_mask = (rgb[:, :, 0] > 220) & (rgb[:, :, 1] > 220) & (rgb[:, :, 2] > 220)
    black_mask = (rgb[:, :, 0] < 35) & (rgb[:, :, 1] < 35) & (rgb[:, :, 2] < 35)
    ratios = {
        "white_pixel_ratio": float(white_mask.mean()),
        "black_pixel_ratio": float(black_mask.mean()),
        "bright_pixel_ratio": float((gray_norm > 0.75).mean()),
        "dark_pixel_ratio": float((gray_norm < 0.25).mean()),
        "low_saturation_ratio": float((saturation < 0.2).mean()),
        "high_saturation_ratio": float((saturation > 0.65).mean()),
        "high_value_ratio": float((value > 0.75).mean()),
        "low_value_ratio": float((value < 0.25).mean()),
    }
    return list(ratios.values()), list(ratios.keys())


def _glcm_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    small = cv2.resize(gray.astype(np.uint8), (96, 96), interpolation=cv2.INTER_AREA)
    quantized = (small // 32).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=8,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats: list[float] = []
    names: list[str] = []
    for prop in props:
        values = graycoprops(glcm, prop).astype(np.float32)
        feats.extend([float(values.mean()), float(values.std())])
        names.extend([f"glcm_{prop}_mean", f"glcm_{prop}_std"])
    return feats, names


def _lbp_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    small = cv2.resize(gray.astype(np.uint8), (128, 128), interpolation=cv2.INTER_AREA)
    feats: list[float] = []
    names: list[str] = []
    for radius, points in ((1, 8), (2, 16), (3, 24)):
        lbp = local_binary_pattern(small, P=points, R=radius, method="uniform")
        bins = points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        feats.extend(hist.astype(np.float32).tolist())
        names.extend([f"lbp_r{radius}_uniform_{idx:02d}" for idx in range(bins)])
    return feats, names


def _gabor_features(gray: np.ndarray) -> tuple[list[float], list[str]]:
    small = cv2.resize(gray.astype(np.float32) / 255.0, (128, 128), interpolation=cv2.INTER_AREA)
    feats: list[float] = []
    names: list[str] = []
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lambdas = [4.0, 8.0]
    for lambda_idx, wavelength in enumerate(lambdas, start=1):
        for angle_idx, theta in enumerate(angles):
            kernel = cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=4.0,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(small, cv2.CV_32F, kernel)
            abs_response = np.abs(response)
            prefix = f"gabor_l{lambda_idx}_a{angle_idx}"
            feats.extend(
                [
                    float(abs_response.mean()),
                    float(abs_response.std()),
                    float(np.mean(abs_response ** 2)),
                ]
            )
            names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_energy"])
    return feats, names


def _specular_features(gray: np.ndarray, hsv: np.ndarray, edges: np.ndarray) -> tuple[list[float], list[str]]:
    gray_norm = gray.astype(np.float32) / 255.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    highlight_mask = (value > 0.85) & (saturation < 0.25)
    strong_highlight_mask = (value > 0.92) & (saturation < 0.18)
    edge_mask = edges > 0
    if highlight_mask.any():
        highlight_gray = gray_norm[highlight_mask]
        highlight_contrast = float(highlight_gray.std())
        specular_edge_ratio = float((edge_mask & highlight_mask).sum() / max(int(highlight_mask.sum()), 1))
    else:
        highlight_contrast = 0.0
        specular_edge_ratio = 0.0
    feats = [
        float(highlight_mask.mean()),
        float(strong_highlight_mask.mean()),
        float(((value > 0.75) & (saturation < 0.20)).mean()),
        specular_edge_ratio,
        highlight_contrast,
    ]
    names = [
        "highlight_ratio",
        "strong_highlight_ratio",
        "low_sat_high_value_ratio",
        "specular_edge_ratio",
        "highlight_contrast_std",
    ]
    return feats, names


def _spatial_grid_features(gray: np.ndarray, hsv: np.ndarray, edges: np.ndarray, grid_size: int = 3) -> tuple[list[float], list[str]]:
    h, w = gray.shape
    gray_norm = gray.astype(np.float32) / 255.0
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    edge_mask = edges > 0
    feats: list[float] = []
    names: list[str] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * h // grid_size
            y1 = (row + 1) * h // grid_size
            x0 = col * w // grid_size
            x1 = (col + 1) * w // grid_size
            cell_gray = gray_norm[y0:y1, x0:x1]
            cell_sat = saturation[y0:y1, x0:x1]
            cell_edges = edge_mask[y0:y1, x0:x1]
            prefix = f"grid{grid_size}_{row}_{col}"
            feats.extend(
                [
                    float(cell_gray.mean()),
                    float(cell_gray.std()),
                    float(cell_sat.mean()),
                    float(cell_edges.mean()),
                ]
            )
            names.extend(
                [
                    f"{prefix}_brightness_mean",
                    f"{prefix}_brightness_std",
                    f"{prefix}_saturation_mean",
                    f"{prefix}_edge_density",
                ]
            )
    return feats, names


def _foreground_shape_features(gray: np.ndarray, hsv: np.ndarray) -> tuple[list[float], list[str]]:
    gray_u8 = gray.astype(np.uint8)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    _, otsu = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fg = gray_u8 < np.percentile(gray_u8, 70)
    sat_fg = saturation > 0.18
    mask = ((otsu > 0) & sat_fg) | dark_fg
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    image_area = float(h * w)
    names = [
        "foreground_area_ratio",
        "foreground_bbox_area_ratio",
        "foreground_bbox_aspect_ratio",
        "foreground_bbox_fill_ratio",
        "foreground_centroid_x",
        "foreground_centroid_y",
        "foreground_perimeter_ratio",
        "foreground_circularity",
        "foreground_num_contours",
    ]
    if not contours:
        return [0.0] * len(names), names

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    x, y, bw, bh = cv2.boundingRect(largest)
    perimeter = float(cv2.arcLength(largest, True))
    moments = cv2.moments(largest)
    if moments["m00"] > 1e-8:
        cx = float(moments["m10"] / moments["m00"]) / w
        cy = float(moments["m01"] / moments["m00"]) / h
    else:
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
    bbox_area = float(bw * bh)
    circularity = 0.0 if perimeter <= 1e-8 else float(4.0 * np.pi * area / (perimeter ** 2))
    feats = [
        area / image_area,
        bbox_area / image_area,
        float(bw / max(bh, 1)),
        area / max(bbox_area, 1.0),
        cx,
        cy,
        perimeter / max(2.0 * (h + w), 1.0),
        circularity,
        float(len(contours)),
    ]
    return feats, names


def _dominant_color_features(rgb: np.ndarray, n_colors: int = 3) -> tuple[list[float], list[str]]:
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32) / 255.0
    if len(np.unique(pixels, axis=0)) < n_colors:
        centers = np.zeros((n_colors, 3), dtype=np.float32)
        ratios = np.zeros(n_colors, dtype=np.float32)
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        for out_idx, src_idx in enumerate(order[:n_colors]):
            centers[out_idx] = unique[src_idx]
            ratios[out_idx] = counts[src_idx] / pixels.shape[0]
    else:
        km = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=42,
            batch_size=1024,
            n_init=3,
            max_iter=50,
        )
        labels = km.fit_predict(pixels)
        counts = np.bincount(labels, minlength=n_colors).astype(np.float32)
        ratios = counts / max(float(counts.sum()), 1.0)
        order = np.argsort(ratios)[::-1]
        centers = km.cluster_centers_[order]
        ratios = ratios[order]

    feats: list[float] = []
    names: list[str] = []
    for idx in range(n_colors):
        feats.extend(centers[idx].astype(np.float32).tolist())
        feats.append(float(ratios[idx]))
        names.extend(
            [
                f"dominant_color_{idx + 1}_r",
                f"dominant_color_{idx + 1}_g",
                f"dominant_color_{idx + 1}_b",
                f"dominant_color_{idx + 1}_ratio",
            ]
        )
    return feats, names


def extract_handcrafted_features(path: str | Path, image_size: int = 224) -> tuple[np.ndarray, list[str]]:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    rgb_float = rgb.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    features: list[float] = []
    names: list[str] = []

    for arr, prefix in ((rgb_float, "rgb"), (hsv, "hsv"), (lab, "lab")):
        vals, cols = _channel_stats(arr, prefix)
        features.extend(vals)
        names.extend(cols)
        vals, cols = _color_moments(arr, prefix)
        features.extend(vals)
        names.extend(cols)

    brightness = gray / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(rgb, 100, 200)
    saturation = hsv[:, :, 1] / 255.0

    extra = {
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "brightness_min": float(brightness.min()),
        "brightness_max": float(brightness.max()),
        "contrast_gray_std": float(gray.std()),
        "contrast_rms": float(np.sqrt(np.mean((brightness - brightness.mean()) ** 2))),
        "sharpness_laplacian_var": float(lap.var()),
        "edge_density": float((edges > 0).mean()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
    }
    names.extend(extra.keys())
    features.extend(extra.values())

    vals, cols = _ratio_features(rgb, gray, hsv)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _entropy_from_uint8(gray.astype(np.uint8), "gray")
    features.extend(vals)
    names.extend(cols)
    for idx, channel_name in enumerate(("r", "g", "b")):
        vals, cols = _entropy_from_uint8(rgb[:, :, idx], f"{channel_name}_channel")
        features.extend(vals)
        names.extend(cols)

    vals, cols = _glcm_features(gray)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _lbp_features(gray)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _gabor_features(gray)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _specular_features(gray, hsv, edges)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _spatial_grid_features(gray, hsv, edges, grid_size=3)
    features.extend(vals)
    names.extend(cols)

    vals, cols = _foreground_shape_features(gray, hsv)
    features.extend(vals)
    names.extend(cols)

    hist_features = []
    hist_names = []
    for idx, channel_name in enumerate(("r", "g", "b")):
        hist, _ = np.histogram(rgb[:, :, idx], bins=16, range=(0, 256), density=True)
        hist_features.extend(hist.astype(np.float32).tolist())
        hist_names.extend([f"hist_{channel_name}_{bin_idx:02d}" for bin_idx in range(16)])
    names.extend(hist_names)
    features.extend(hist_features)

    vals, cols = _dominant_color_features(rgb, n_colors=3)
    features.extend(vals)
    names.extend(cols)

    return np.asarray(features, dtype=np.float32), names


def build_feature_matrix(rows, image_size: int = 224):
    X, y, paths = [], [], []
    feature_names: list[str] | None = None
    for idx, (path, label, _) in enumerate(rows, start=1):
        feats, names = extract_handcrafted_features(path, image_size=image_size)
        if feature_names is None:
            feature_names = names
        X.append(feats)
        y.append(label)
        paths.append(str(path))
        if idx % 500 == 0:
            print(f"Extracted {idx}/{len(rows)} images")
    return np.vstack(X), np.asarray(y, dtype=np.int64), paths, feature_names or []


def save_npz(output_path: Path, arrays: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def train_models(X_train, y_train, X_val, y_val, label_map, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    for name, model in models.items():
        print(f"\n=== Train {name} ===")
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        acc = accuracy_score(y_val, pred)
        print(f"val_accuracy: {acc:.4f}")
        print(classification_report(y_val, pred, target_names=target_names, digits=4))
        print("confusion_matrix:")
        print(confusion_matrix(y_val, pred))
        joblib.dump(model, output_dir / f"{name}.joblib")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract handcrafted features and train ML models.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR / "handcrafted")
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-augmented-train", action="store_true")
    parser.add_argument("--train-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")

    train_split = "train_augmented" if args.use_augmented_train else "train"
    train_rows = list_images(args.split_dir, train_split, label_map)
    val_rows = list_images(args.split_dir, "val", label_map)
    test_rows = list_images(args.split_dir, "test", label_map)

    print(f"train rows ({train_split}): {len(train_rows)}")
    print(f"val rows               : {len(val_rows)}")
    print(f"test rows              : {len(test_rows)}")

    X_train, y_train, train_paths, feature_names = build_feature_matrix(
        train_rows, image_size=args.image_size
    )
    X_val, y_val, val_paths, _ = build_feature_matrix(val_rows, image_size=args.image_size)

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_paths": np.asarray(train_paths),
        "val_paths": np.asarray(val_paths),
    }
    if test_rows:
        X_test, y_test, test_paths, _ = build_feature_matrix(test_rows, image_size=args.image_size)
        arrays.update({"X_test": X_test, "y_test": y_test, "test_paths": np.asarray(test_paths)})

    output_path = args.output_dir / "handcrafted_features.npz"
    save_npz(output_path, arrays)
    with open(args.output_dir / "handcrafted_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"\nSaved features -> {output_path}")

    if args.train_models:
        train_models(X_train, y_train, X_val, y_val, label_map, args.model_dir)


if __name__ == "__main__":
    main()
