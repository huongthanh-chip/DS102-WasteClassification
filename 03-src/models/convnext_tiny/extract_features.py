#!/usr/bin/env python3
"""
Rút trích CNN features từ ConvNeXt-Tiny (best.pt).

Pipeline:
  1. Load ConvNeXt-Tiny đã train, bỏ Linear head cuối (giữ lại 768-dim features)
  2. Cho ảnh qua backbone → vector đặc trưng 768 chiều mỗi ảnh
  3. Lưu features + nhãn ra file .npy để train RF/SVM ở bước tiếp theo

Lý do cần cả train lẫn val:
  - cnn_train.npy : features của tập train_augmented → dùng để TRAIN RF/SVM
  - cnn_val.npy   : features của tập val             → dùng để ĐÁNH GIÁ RF/SVM
  RF/SVM không nhìn ảnh gốc, chỉ học trên vector đặc trưng do ConvNeXt rút ra.
  Val set vẫn là tập chưa từng train → đảm bảo đánh giá khách quan.

Output:
  feature_cache/cnn_train.npy  — shape (N_train, 768)
  feature_cache/cnn_val.npy    — shape (N_val,   768)
  feature_cache/y_train.npy    — nhãn int
  feature_cache/y_val.npy      — nhãn int
  feature_cache/class_names.json

Chạy:
    python extract_features.py
    python extract_features.py --checkpoint checkpoints/best.pt --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT      = Path(__file__).resolve().parents[3]
SRC_ROOT          = PROJECT_ROOT / "03-src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20" / "train_augmented"
DEFAULT_VAL_DIR   = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20" / "val"
DEFAULT_CKPT      = PROJECT_ROOT / "05-models" / "convnext_tiny" / "best.pt"
CACHE_DIR         = PROJECT_ROOT / "04-features" / "convnext_tiny"

# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #

def build_backbone(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """
    Load ConvNeXt-Tiny từ checkpoint, bỏ Linear head cuối.
    Kiến trúc classifier gốc: LayerNorm → Flatten → Linear(768 → num_classes)
    Ta giữ: LayerNorm → Flatten  →  output (B, 768)
    """
    if checkpoint_path.exists():
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = convnext_tiny(weights=None)
        in_f  = model.classifier[2].in_features          # 768
        model.classifier[2] = nn.Linear(in_f, num_classes)
        model.load_state_dict(ckpt["model"])
        print(f"[backbone] Loaded {checkpoint_path}  (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"[backbone] Checkpoint not found — dùng ImageNet pretrained")
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # Bỏ Linear cuối, giữ LayerNorm + Flatten
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Trả về (features, labels) cho toàn bộ loader.
    features shape: (N, 768)
    labels   shape: (N,)
    """
    all_feats, all_labels = [], []
    total = len(loader)

    for i, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        feats  = model(images)                       # (B, 768)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}]", flush=True)

    return (
        np.concatenate(all_feats,  axis=0),
        np.concatenate(all_labels, axis=0),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Transform — không augment, chỉ resize + normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.6320, 0.6092, 0.5805),
                             std =(0.2012, 0.1991, 0.2094)),
    ])

    # Dataset
    train_ds = ImageFolder(str(args.train_dir), transform=transform)
    val_ds   = ImageFolder(str(args.val_dir),   transform=transform)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)} ảnh  |  Val: {len(val_ds)} ảnh\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Backbone
    backbone = build_backbone(Path(args.checkpoint), num_classes, device)

    # Extract
    print("\nExtracting train features ...")
    X_train, y_train = extract(backbone, train_loader, device)
    print(f"  → shape: {X_train.shape}")

    print("\nExtracting val features ...")
    X_val, y_val = extract(backbone, val_loader, device)
    print(f"  → shape: {X_val.shape}")

    # Lưu
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "cnn_train.npy", X_train)
    np.save(CACHE_DIR / "cnn_val.npy",   X_val)
    np.save(CACHE_DIR / "y_train.npy",   y_train)
    np.save(CACHE_DIR / "y_val.npy",     y_val)
    (CACHE_DIR / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2))

    print(f"\nĐã lưu vào {CACHE_DIR}/")
    print("  cnn_train.npy  cnn_val.npy  y_train.npy  y_val.npy  class_names.json")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rút trích CNN features từ ConvNeXt-Tiny")
    p.add_argument("--checkpoint",  type=str,  default=str(DEFAULT_CKPT))
    p.add_argument("--train-dir",   type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--val-dir",     type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--batch-size",  type=int,  default=64)
    p.add_argument("--num-workers", type=int,  default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
