#!/usr/bin/env python3
"""
Train ConvNeXt_Tiny — Phân loại rác thải 8 class.

Chạy training:
    python ConvNeXt_Tiny.py --use-augmented-dir
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Paths — chỉnh lại cho phù hợp với cấu trúc thư mục của bạn
# =============================================================================

PROJECT_ROOT             = Path(__file__).resolve().parents[3]
SRC_ROOT                 = PROJECT_ROOT / "03-src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_SPLIT_DIR        = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_AUGMENTED_TRAIN_DIR = DEFAULT_SPLIT_DIR / "train_augmented"
DEFAULT_LABEL_MAP        = PROJECT_ROOT / "04-features" / "label_map.json"

CHECKPOINT_DIR = PROJECT_ROOT / "05-models" / "convnext_tiny"
LOG_DIR        = CHECKPOINT_DIR / "logs"

# =============================================================================
# Dataset constants
# =============================================================================

IMAGE_EXTS   = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TARGET_SIZE  = 224
DATASET_MEAN = (0.6320, 0.6092, 0.5805)
DATASET_STD  = (0.2012, 0.1991, 0.2094)
RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.25

# =============================================================================
# Training hyper-params
# =============================================================================

EARLY_STOPPING_PATIENCE  = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
REDUCE_LR_PATIENCE       = 3
REDUCE_LR_FACTOR         = 0.5


# =============================================================================
# Data utilities (inlined từ dataloader.py)
# =============================================================================

def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def stratified_train_val_split(
    paths: list[Path],
    labels: list[int],
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    counts   = Counter(labels)
    too_small = [label for label, count in counts.items() if count < 2]
    stratify  = labels if not too_small else None
    if too_small:
        print("[WARN] Some classes have fewer than 2 images; falling back to non-stratified split.")
    return train_test_split(paths, labels, test_size=val_size, random_state=seed, stratify=stratify)


def make_transforms(
    image_size: int = TARGET_SIZE,
    mean: tuple = DATASET_MEAN,
    std: tuple  = DATASET_STD,
    augment_train: bool = True,
) -> tuple[Callable, Callable]:
    train_steps: list[Callable] = [T.Resize((image_size, image_size))]
    if augment_train:
        train_steps.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=12),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
        ])
    train_steps.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return T.Compose(train_steps), val_transform


class ImagePathDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str | Path],
        labels: Iterable[int],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.paths      = [Path(p) for p in paths]
        self.labels     = [int(y) for y in labels]
        self.transform  = transform
        self.return_path = return_path
        if len(self.paths) != len(self.labels):
            raise ValueError("paths and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path  = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(path)
        return image, label


def compute_class_weights(labels: list[int], num_classes: int) -> np.ndarray:
    counts   = np.bincount(np.array(labels, dtype=int), minlength=num_classes)
    total    = counts.sum()
    weights  = np.zeros(num_classes, dtype=np.float32)
    nonzero  = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights


def make_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    class_weights  = compute_class_weights(labels, num_classes)
    sample_weights = [float(class_weights[label]) for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _make_dataloaders_from_paths(
    train_paths: list[Path],
    val_paths: list[Path],
    train_labels: list[int],
    val_labels: list[int],
    label_map: dict[str, int],
    data_source: str | Path,
    batch_size: int = 32,
    image_size: int = TARGET_SIZE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    train_transform, val_transform = make_transforms(image_size=image_size, augment_train=augment_train)
    train_dataset = ImagePathDataset(train_paths, train_labels, transform=train_transform)
    val_dataset   = ImagePathDataset(val_paths,   val_labels,   transform=val_transform)

    num_classes = len(label_map)
    sampler     = make_weighted_sampler(train_labels, num_classes) if use_weighted_sampler else None
    shuffle     = sampler is None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    info = {
        "data_dir"    : str(Path(data_source)),
        "label_map"   : label_map,
        "idx_to_class": {idx: cls for cls, idx in label_map.items()},
        "num_classes" : num_classes,
        "train_size"  : len(train_dataset),
        "val_size"    : len(val_dataset),
        "train_counts": dict(sorted(Counter(train_labels).items())),
        "val_counts"  : dict(sorted(Counter(val_labels).items())),
        "class_weights": compute_class_weights(train_labels, num_classes),
        "image_size"  : image_size,
        "mean"        : DATASET_MEAN,
        "std"         : DATASET_STD,
    }
    return train_loader, val_loader, info


def build_dataloaders_from_train_val_dirs(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int = 32,
    image_size: int = TARGET_SIZE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    train_dir  = Path(train_dir)
    val_dir    = Path(val_dir)
    label_map  = load_label_map(label_map_path) or discover_label_map(train_dir)
    train_paths, train_labels, label_map = scan_image_folder(train_dir, label_map=label_map)
    val_paths,   val_labels,   _         = scan_image_folder(val_dir,   label_map=label_map)
    return _make_dataloaders_from_paths(
        train_paths=train_paths, val_paths=val_paths,
        train_labels=train_labels, val_labels=val_labels,
        label_map=label_map, data_source=train_dir.parent,
        batch_size=batch_size, image_size=image_size,
        num_workers=num_workers, augment_train=augment_train,
        use_weighted_sampler=use_weighted_sampler, pin_memory=pin_memory,
    )


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_size: float = DEFAULT_VAL_SIZE,
    image_size: int = TARGET_SIZE,
    seed: int = RANDOM_STATE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    seed_everything(seed)
    data_dir = Path(data_dir)

    if (data_dir / "train").exists() and (data_dir / "val").exists():
        # data_dir là split folder
        label_map    = load_label_map(label_map_path) or discover_label_map(data_dir / "train")
        train_paths, train_labels, label_map = scan_image_folder(data_dir / "train", label_map=label_map)
        val_paths,   val_labels,   _         = scan_image_folder(data_dir / "val",   label_map=label_map)
        return _make_dataloaders_from_paths(
            train_paths=train_paths, val_paths=val_paths,
            train_labels=train_labels, val_labels=val_labels,
            label_map=label_map, data_source=data_dir,
            batch_size=batch_size, image_size=image_size,
            num_workers=num_workers, augment_train=augment_train,
            use_weighted_sampler=use_weighted_sampler, pin_memory=pin_memory,
        )

    paths, labels, label_map = scan_image_folder(data_dir, label_map_path=label_map_path)
    train_paths, val_paths, train_labels, val_labels = stratified_train_val_split(
        paths, labels, val_size=val_size, seed=seed
    )
    return _make_dataloaders_from_paths(
        train_paths=train_paths, val_paths=val_paths,
        train_labels=train_labels, val_labels=val_labels,
        label_map=label_map, data_source=data_dir,
        batch_size=batch_size, image_size=image_size,
        num_workers=num_workers, augment_train=augment_train,
        use_weighted_sampler=use_weighted_sampler, pin_memory=pin_memory,
    )


# =============================================================================
# Model
# =============================================================================

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load ConvNeXt_Tiny pretrained trên ImageNet, thay classifier head.

    Tại sao dùng pretrained?
      ConvNeXt đã học được đặc trưng cơ bản (cạnh, texture, hình dạng) từ
      1.2 triệu ảnh ImageNet. Fine-tune trên data rác giúp model hội tụ
      nhanh hơn và cần ít data hơn so với train từ đầu.

    Thay layer cuối:
      classifier[2]: Linear(768 → 1000)  ← ImageNet 1000 class
                   → Linear(768 → 8)     ← 8 class rác của mình
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model   = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features  # 768
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# =============================================================================
# Helpers
# =============================================================================

class AverageMeter:
    """Tính trung bình loss/accuracy qua từng batch trong 1 epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """% ảnh dự đoán đúng trong 1 batch."""
    return (output.argmax(dim=1) == target).float().mean().item() * 100.0


# =============================================================================
# Train / Validate
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, total_epochs):
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        n = images.size(0)
        loss_m.update(loss.item(), n)
        acc_m.update(accuracy(logits.detach(), labels), n)

        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            print(
                f"  Ep"
                f"och [{epoch}/{total_epochs}] Step [{step+1}/{len(loader)}]"
                f"  Loss: {loss_m.avg:.4f}  Acc: {acc_m.avg:.2f}%",
                flush=True,
            )

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Đánh giá trên tập val — KHÔNG cập nhật trọng số.
    Trả về val_loss, val_acc, val_macro_f1.
    """
    model.eval()
    loss_m     = AverageMeter()
    acc_m      = AverageMeter()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        n      = images.size(0)
        loss_m.update(loss.item(), n)
        acc_m.update(accuracy(logits, labels), n)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return loss_m.avg, acc_m.avg, val_macro_f1


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved -> {path}")


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  [ckpt] Resumed from {path}  (epoch {ckpt.get('epoch', '?')})")
    return (
        ckpt.get("epoch", 0),
        ckpt.get("best_val_f1", 0.0),
        ckpt.get("best_val_acc", 0.0),
    )


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Device: {device}  (AMP: {use_amp})")

    # ── 1. Dataloader ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    if args.use_augmented_dir:
        train_loader, val_loader, info = build_dataloaders_from_train_val_dirs(
            train_dir            = DEFAULT_AUGMENTED_TRAIN_DIR,
            val_dir              = DEFAULT_SPLIT_DIR / "val",
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            augment_train        = False,
            use_weighted_sampler = args.weighted_sampler,
            label_map_path       = DEFAULT_LABEL_MAP,
        )
    else:
        train_loader, val_loader, info = build_dataloaders(
            data_dir             = args.data_dir,
            batch_size           = args.batch_size,
            val_size             = args.val_size,
            num_workers          = args.num_workers,
            augment_train        = True,
            use_weighted_sampler = args.weighted_sampler,
            label_map_path       = DEFAULT_LABEL_MAP,
        )

    num_classes = info["num_classes"]
    label_map   = info["label_map"]
    print(f"Classes : {num_classes} → {list(label_map.keys())}")
    print(f"Train   : {info['train_size']} ảnh")
    print(f"Val     : {info['val_size']} ảnh")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\nBuilding ConvNeXt-Tiny...")
    model = build_model(num_classes=num_classes, pretrained=not args.from_scratch)
    model = model.to(device)
    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Loss function ──────────────────────────────────────────────────────
    if args.weighted_loss:
        class_weights = torch.tensor(info["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Loss    : weighted CrossEntropyLoss (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"Loss    : CrossEntropyLoss (smoothing={args.label_smoothing})")

    # ── 4. Optimizer ──────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = REDUCE_LR_FACTOR,
        patience = REDUCE_LR_PATIENCE,
        min_lr   = args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_f1  = 0.0
    best_val_acc = 0.0
    if args.resume:
        start_epoch, best_val_f1, best_val_acc = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        start_epoch += 1

    # ── 5. Training loop ──────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "train_log.json"
    history: list[dict] = []

    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Training ConvNeXt-Tiny — max {args.epochs} epochs")
    print(f"  Early stopping : patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
    print(f"  ReduceLR       : patience={REDUCE_LR_PATIENCE}, factor={REDUCE_LR_FACTOR}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs
        )
        val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        is_best = val_macro_f1 > best_val_f1 + EARLY_STOPPING_MIN_DELTA
        if is_best:
            best_val_f1  = val_macro_f1
            best_val_acc = val_acc
            no_improve   = 0
        else:
            no_improve += 1

        print(
            f"\nEpoch {epoch:3d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% F1 {val_macro_f1:.4f} | "
            f"{elapsed:.1f}s" + ("  ★ Best!" if is_best else f"  (no improve: {no_improve}/{EARLY_STOPPING_PATIENCE})"),
            flush=True,
        )
        print()

        history.append({
            "epoch"       : epoch,
            "train_loss"  : round(train_loss,    5),
            "train_acc"   : round(train_acc,     3),
            "val_loss"    : round(val_loss,      5),
            "val_acc"     : round(val_acc,       3),
            "val_macro_f1": round(val_macro_f1,  5),
            "best_val_f1" : round(best_val_f1,   5),
            "lr"          : optimizer.param_groups[0]["lr"],
        })
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        state = {
            "epoch"       : epoch,
            "model"       : model.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "scheduler"   : scheduler.state_dict(),
            "best_val_f1" : best_val_f1,
            "best_val_acc": best_val_acc,
            "label_map"   : label_map,
        }
        save_checkpoint(state, CHECKPOINT_DIR / "last.pt")
        if is_best:
            save_checkpoint(state, CHECKPOINT_DIR / "best.pt")

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping tại epoch {epoch} — val_macro_f1 không cải thiện sau {EARLY_STOPPING_PATIENCE} epoch.")
            break

    print(f"\nDone! Best val F1: {best_val_f1:.5f} | Best val acc: {best_val_acc:.2f}%")
    print(f"Checkpoint tốt nhất: {CHECKPOINT_DIR / 'best.pt'}")
    print(f"Log training       : {log_path}")


# =============================================================================
# Inference
# =============================================================================

def predict_single(image_path: str | Path, checkpoint: str | Path) -> str:
    """Dự đoán class cho 1 ảnh mới dùng best.pt."""
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt         = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_map    = ckpt["label_map"]
    idx_to_class = {v: k for k, v in label_map.items()}

    model = build_model(num_classes=len(label_map), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    print(f"Prediction: {idx_to_class[pred]}  ({probs[pred].item():.1%})")
    return idx_to_class[pred]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ConvNeXt-Tiny — waste classification")

    p.add_argument("--data-dir",          type=Path,  default=Path("Cleaned_Dataset/train"))
    p.add_argument("--val-size",          type=float, default=0.2)
    p.add_argument("--use-augmented-dir", action="store_true",
                   help="Dùng Prepared_Clean_Split/train_augmented (đã augment sẵn)")
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight-decay",      type=float, default=5e-4)
    p.add_argument("--label-smoothing",   type=float, default=0.1)
    p.add_argument("--num-workers",       type=int,   default=0)
    p.add_argument("--from-scratch",      action="store_true",
                   help="Không dùng ImageNet pretrained")
    p.add_argument("--weighted-sampler",  action="store_true",
                   help="WeightedRandomSampler — lấy mẫu cân bằng class")
    p.add_argument("--weighted-loss",     action="store_true",
                   help="Weighted CrossEntropyLoss — phạt nặng hơn class ít ảnh")
    p.add_argument("--resume",            type=str,   default=None,
                   help="Path checkpoint để tiếp tục training")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
