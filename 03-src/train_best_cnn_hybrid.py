#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from tqdm import tqdm

from dataloader import DATASET_MEAN, DATASET_STD, IMAGE_EXTS, TARGET_SIZE, load_label_map

ImageFile.LOAD_TRUNCATED_IMAGES = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "04-features" / "label_map.json"
DEFAULT_HANDCRAFTED = PROJECT_ROOT / "04-features" / "handcrafted_features.npz"
DEFAULT_CONVNEXT_CHECKPOINT = PROJECT_ROOT / "05-models" / "convnext_tiny" / "best.pt"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "04-features" / "best_cnn_hybrid"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "05-models" / "best_cnn_hybrid"


def list_images(split_dir: Path, split: str, label_map: dict[str, int]):
    rows = []
    split_root = split_dir / split
    for class_name, label in sorted(label_map.items(), key=lambda item: item[1]):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append((path.resolve(), label, class_name))
    return rows


class ImageRowsDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, _ = self.rows[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(label), str(path)


def build_convnext_embedding_model(checkpoint_path: Path, num_classes: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model"])
    model.classifier[2] = nn.Identity()
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def extract_split_embeddings(rows, model, batch_size: int, num_workers: int, device: torch.device):
    transform = T.Compose(
        [
            T.Resize((TARGET_SIZE, TARGET_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )
    dataset = ImageRowsDataset(rows, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    features = []
    labels = []
    paths = []
    for images, batch_labels, batch_paths in tqdm(loader, desc="ConvNeXt embeddings"):
        images = images.to(device, non_blocking=True)
        emb = model(images).detach().cpu().numpy().astype(np.float32)
        features.append(emb)
        labels.extend(batch_labels.numpy().astype(np.int64).tolist())
        paths.extend(list(batch_paths))
    return np.vstack(features), np.asarray(labels, dtype=np.int64), np.asarray(paths)


def load_handcrafted_by_path(handcrafted_path: Path, split: str):
    data = np.load(handcrafted_path, allow_pickle=True)
    X = data[f"X_{split}"]
    y = data[f"y_{split}"]
    paths = np.asarray(data[f"{split}_paths"]).astype(str)
    return {str(Path(path).resolve()): idx for idx, path in enumerate(paths)}, X, y


def align_hybrid_features(cnn_arrays: dict, handcrafted_path: Path):
    hybrid = {}
    for split in ("train", "val", "test"):
        cnn_X = cnn_arrays[f"X_{split}_cnn"]
        cnn_y = cnn_arrays[f"y_{split}"]
        cnn_paths = np.asarray(cnn_arrays[f"{split}_paths"]).astype(str)
        hand_index, hand_X, hand_y = load_handcrafted_by_path(handcrafted_path, split)

        hand_indices = []
        missing = []
        for path in cnn_paths:
            idx = hand_index.get(str(Path(path).resolve()))
            if idx is None:
                missing.append(path)
            else:
                hand_indices.append(idx)
        if missing:
            raise ValueError(f"{split}: {len(missing)} paths missing in handcrafted features. First: {missing[0]}")

        hand_indices_arr = np.asarray(hand_indices, dtype=np.int64)
        if not np.array_equal(cnn_y, hand_y[hand_indices_arr]):
            raise ValueError(f"{split}: labels do not match after path alignment.")

        hybrid[f"X_{split}"] = np.concatenate([cnn_X, hand_X[hand_indices_arr]], axis=1).astype(np.float32)
        hybrid[f"y_{split}"] = cnn_y.astype(np.int64)
        hybrid[f"{split}_paths"] = cnn_paths
        print(f"{split}: cnn={cnn_X.shape}, handcrafted={hand_X[hand_indices_arr].shape}, hybrid={hybrid[f'X_{split}'].shape}")
    return hybrid


def evaluate_model(model, arrays: dict, idx_to_class: dict[int, str]):
    labels = sorted(idx_to_class)
    target_names = [idx_to_class[idx] for idx in labels]
    metrics = {}
    predictions = {}
    for split in ("val", "test"):
        y_true = arrays[f"y_{split}"]
        y_pred = model.predict(arrays[f"X_{split}"])
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        metrics[split] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }
        predictions[split] = y_pred
    return metrics, predictions


def save_predictions(path: Path, paths, y_true, y_pred, idx_to_class: dict[int, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "true_label", "true_class", "pred_label", "pred_class", "correct"],
        )
        writer.writeheader()
        for img_path, true, pred in zip(paths, y_true, y_pred):
            writer.writerow(
                {
                    "path": str(img_path),
                    "true_label": int(true),
                    "true_class": idx_to_class[int(true)],
                    "pred_label": int(pred),
                    "pred_class": idx_to_class[int(pred)],
                    "correct": int(int(true) == int(pred)),
                }
            )


def train_hybrid_classifiers(arrays: dict, label_map: dict[str, int], model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    models = {
        "softmax_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=10000,
                        dual="auto",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        ),
    }

    all_metrics = {}
    for name, model in models.items():
        print(f"\n=== Train {name} ===")
        model.fit(arrays["X_train"], arrays["y_train"])
        joblib.dump(model, model_dir / f"{name}.joblib")
        metrics, predictions = evaluate_model(model, arrays, idx_to_class)
        all_metrics[name] = metrics
        for split in ("val", "test"):
            report = metrics[split]["classification_report"]
            print(
                f"{split}: acc={metrics[split]['accuracy']:.4f} "
                f"macro_f1={report['macro avg']['f1-score']:.4f} "
                f"weighted_f1={report['weighted avg']['f1-score']:.4f}"
            )
            save_predictions(
                model_dir / f"{name}_{split}_predictions.csv",
                arrays[f"{split}_paths"],
                arrays[f"y_{split}"],
                predictions[split],
                idx_to_class,
            )

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved hybrid models and metrics -> {model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use the best CNN backbone features with handcrafted features for RF/SVM/Softmax baselines."
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--handcrafted", type=Path, default=DEFAULT_HANDCRAFTED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONVNEXT_CHECKPOINT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")

    args.feature_dir.mkdir(parents=True, exist_ok=True)
    cnn_path = args.feature_dir / "convnext_tiny_cnn_features.npz"
    hybrid_path = args.feature_dir / "convnext_tiny_hybrid_features.npz"

    if args.skip_extract and cnn_path.exists():
        cnn_arrays = dict(np.load(cnn_path, allow_pickle=True))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Best CNN selected: ConvNeXt-Tiny")
        print("Reason: current recorded test macro F1 is about 0.9344, higher than EfficientNet-B0 0.9191.")
        print(f"Device: {device}")
        model, checkpoint = build_convnext_embedding_model(args.checkpoint, len(label_map), device)
        print(f"Loaded ConvNeXt checkpoint: {args.checkpoint} epoch={checkpoint.get('epoch', '?')}")

        cnn_arrays = {}
        for split in ("train_augmented", "val", "test"):
            out_split = "train" if split == "train_augmented" else split
            rows = list_images(args.split_dir, split, label_map)
            print(f"\nExtract {out_split}: {len(rows)} images")
            X, y, paths = extract_split_embeddings(rows, model, args.batch_size, args.num_workers, device)
            cnn_arrays[f"X_{out_split}_cnn"] = X
            cnn_arrays[f"y_{out_split}"] = y
            cnn_arrays[f"{out_split}_paths"] = paths
        np.savez_compressed(cnn_path, **cnn_arrays)
        print(f"Saved CNN features -> {cnn_path}")

    hybrid_arrays = align_hybrid_features(cnn_arrays, args.handcrafted)
    np.savez_compressed(hybrid_path, **hybrid_arrays)
    print(f"Saved hybrid features -> {hybrid_path}")

    if not args.skip_train:
        train_hybrid_classifiers(hybrid_arrays, label_map, args.model_dir)


if __name__ == "__main__":
    main()
