#!/usr/bin/env python3
"""
Evaluate model trên tập val.

Chạy sau khi train xong:
    python evaluate.py

Kết quả:
    - Accuracy
    - F1, Precision, Recall từng class
    - Confusion matrix
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image, ImageFile
import json
from typing import Callable, Iterable

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT      = Path(__file__).resolve().parents[3]
SRC_ROOT          = PROJECT_ROOT / "03-src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
CHECKPOINT_DIR    = PROJECT_ROOT / "05-models" / "convnext_tiny"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "04-features" / "label_map.json"

# =============================================================================
# Constants
# =============================================================================

TARGET_SIZE  = 224
DATASET_MEAN = (0.6320, 0.6092, 0.5805)
DATASET_STD  = (0.2012, 0.1991, 0.2094)
IMAGE_EXTS   = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# =============================================================================
# Data utilities
# =============================================================================

def load_label_map(path: str | Path = DEFAULT_LABEL_MAP) -> dict[str, int] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {str(k): int(v) for k, v in label_map.items()}


def discover_label_map(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {data_dir}")
    return {class_name: idx for idx, class_name in enumerate(classes)}


def get_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def scan_image_folder(
    data_dir: str | Path,
    label_map: dict[str, int] | None = None,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
) -> tuple[list[Path], list[int], dict[str, int]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    if label_map is None:
        label_map = load_label_map(label_map_path) or discover_label_map(data_dir)

    paths: list[Path] = []
    labels: list[int] = []
    missing_classes: list[str] = []

    for class_name, class_idx in sorted(label_map.items(), key=lambda item: item[1]):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue
        class_paths = get_image_paths(class_dir)
        paths.extend(class_paths)
        labels.extend([class_idx] * len(class_paths))

    if not paths:
        raise ValueError(f"No images found in {data_dir}")
    if missing_classes:
        print(f"[WARN] Missing class folders: {missing_classes}")

    return paths, labels, label_map


class ImagePathDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str | Path],
        labels: Iterable[int],
        transform: Callable | None = None,
    ) -> None:
        self.paths     = [Path(p) for p in paths]
        self.labels    = [int(y) for y in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


# =============================================================================
# Model
# =============================================================================

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model   = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# =============================================================================
# Evaluate
# =============================================================================

def load_val_loader(
    val_dir: Path,
    label_map: dict[str, int],
    batch_size: int = 32,
) -> DataLoader:
    val_paths, val_labels, _ = scan_image_folder(val_dir, label_map=label_map)

    transform = T.Compose([
        T.Resize((TARGET_SIZE, TARGET_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])

    dataset = ImagePathDataset(val_paths, val_labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


def print_results(
    labels: list[int],
    preds: list[int],
    idx_to_class: dict[int, str],
) -> None:
    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

    acc = accuracy_score(labels, preds) * 100
    print(f"\n{'='*60}")
    print(f"  EVALUATE RESULTS — Val Set")
    print(f"{'='*60}")
    print(f"\nAccuracy: {acc:.2f}%\n")

    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix (hàng = thật, cột = dự đoán):")
    header = f"{'':15}" + "".join(f"{c[:8]:>10}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:15}" + "".join(f"{v:>10}" for v in row)
        print(row_str)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    ckpt         = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_map    = ckpt["label_map"]
    idx_to_class = {v: k for k, v in label_map.items()}
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, best_val_acc: {ckpt.get('best_val_acc', '?'):.2f}%")

    model = build_model(num_classes=len(label_map), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    val_dir = DEFAULT_SPLIT_DIR / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy val dir: {val_dir}")

    print(f"Val dir: {val_dir}")
    val_loader = load_val_loader(val_dir, label_map, batch_size=32)
    print(f"Val size: {len(val_loader.dataset)} ảnh")

    print("\nĐang chạy inference...")
    labels, preds = run_inference(model, val_loader, device)
    print_results(labels, preds, idx_to_class)


if __name__ == "__main__":
    main()
