#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SRC = PROJECT_ROOT / "03-src" / "data"
EFFICIENTNET_SRC = PROJECT_ROOT / "03-src" / "models" / "efficientnet_b0"
for src_path in (DATA_SRC, EFFICIENTNET_SRC):
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from dataloader import (
    DATASET_MEAN,
    DATASET_STD,
    IMAGE_EXTS,
    TARGET_SIZE,
    load_label_map,
)
from feature_engineering import extract_handcrafted_features, train_models
from train_efficientnet import create_efficientnet_b0


DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "04-features"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "05-models"
DEFAULT_LABEL_MAP = DEFAULT_FEATURE_DIR / "label_map.json"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "efficientnet_b0" / "best.pt"


class EfficientNetEmbedding(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def collect_split(split_dir: Path, split: str, label_map: dict[str, int]):
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


def load_embedding_model(
    checkpoint_path: Path | None,
    num_classes: int,
    pretrained: bool,
    device: torch.device,
) -> nn.Module:
    dropout = 0.4
    checkpoint = None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        dropout = float(checkpoint.get("args", {}).get("dropout", dropout))

    model = create_efficientnet_b0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded checkpoint -> {checkpoint_path}")
    else:
        print("using EfficientNet-B0 without project checkpoint")
    embedding_model = EfficientNetEmbedding(model).to(device)
    embedding_model.eval()
    return embedding_model


def make_transform(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )


@torch.no_grad()
def extract_cnn_embeddings(rows, model, transform, device, batch_size: int):
    embeddings = []
    labels = []
    paths = []
    batch = []

    def flush():
        if not batch:
            return
        tensors = torch.stack([item[0] for item in batch]).to(device)
        emb = model(tensors).detach().cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        labels.extend([item[1] for item in batch])
        paths.extend([str(item[2]) for item in batch])
        batch.clear()

    for idx, (path, label, _) in enumerate(rows, start=1):
        image = Image.open(path).convert("RGB")
        batch.append((transform(image), int(label), path))
        if len(batch) >= batch_size:
            flush()
        if idx % 500 == 0:
            print(f"CNN embeddings {idx}/{len(rows)}")
    flush()

    return np.vstack(embeddings), np.asarray(labels, dtype=np.int64), np.asarray(paths)


def extract_handcrafted_matrix(rows, image_size: int):
    features = []
    names = None
    for idx, (path, _, _) in enumerate(rows, start=1):
        feat, feat_names = extract_handcrafted_features(path, image_size=image_size)
        features.append(feat)
        names = feat_names
        if idx % 500 == 0:
            print(f"Handcrafted features {idx}/{len(rows)}")
    return np.vstack(features).astype(np.float32), names or []


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CNN and hybrid features.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR / "hybrid")
    parser.add_argument("--image-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-augmented-train", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--skip-handcrafted", action="store_true")
    parser.add_argument("--train-hybrid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    label_map = load_label_map(args.label_map)
    if label_map is None:
        raise FileNotFoundError(f"Label map not found: {args.label_map}")

    train_split = "train_augmented" if args.use_augmented_train else "train"
    splits = {
        "train": collect_split(args.split_dir, train_split, label_map),
        "val": collect_split(args.split_dir, "val", label_map),
        "test": collect_split(args.split_dir, "test", label_map),
    }
    print({name: len(rows) for name, rows in splits.items()})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embedding_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(label_map),
        pretrained=args.pretrained,
        device=device,
    )
    transform = make_transform(args.image_size)

    output = {}
    hybrid = {}
    feature_names = None

    for split_name, rows in splits.items():
        if not rows:
            continue
        X_cnn, y, paths = extract_cnn_embeddings(
            rows, model, transform, device, batch_size=args.batch_size
        )
        output[f"X_{split_name}_cnn"] = X_cnn
        output[f"y_{split_name}"] = y
        output[f"{split_name}_paths"] = paths

        if not args.skip_handcrafted:
            X_hand, feature_names = extract_handcrafted_matrix(rows, image_size=args.image_size)
            X_hybrid = np.concatenate([X_cnn, X_hand], axis=1).astype(np.float32)
            hybrid[f"X_{split_name}"] = X_hybrid
            hybrid[f"y_{split_name}"] = y
            hybrid[f"{split_name}_paths"] = paths

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cnn_path = args.output_dir / "cnn_features_efficientnet_b0.npz"
    np.savez_compressed(cnn_path, **output)
    print(f"saved CNN features -> {cnn_path}")

    if hybrid:
        hybrid_path = args.output_dir / "hybrid_cnn_handcrafted_features.npz"
        np.savez_compressed(hybrid_path, **hybrid)
        print(f"saved hybrid features -> {hybrid_path}")
        if feature_names is not None:
            names = [f"cnn_{idx:04d}" for idx in range(hybrid["X_train"].shape[1] - len(feature_names))]
            names.extend(feature_names)
            with open(args.output_dir / "hybrid_feature_names.json", "w", encoding="utf-8") as f:
                json.dump(names, f, indent=2)

        if args.train_hybrid:
            train_models(
                hybrid["X_train"],
                hybrid["y_train"],
                hybrid["X_val"],
                hybrid["y_val"],
                label_map,
                args.model_dir,
            )


if __name__ == "__main__":
    main()
