#!/usr/bin/env python3
"""
Dataloader for the waste image classification project.

Default input:
  01-data/Merged_Dataset/train/<class_name>/*.png

Main usage:
  from dataloader import build_dataloaders

  train_loader, val_loader, info = build_dataloaders(
      data_dir="01-data/Prepared_Merged_Clean_Split_60_20_20",
      batch_size=32,
      use_weighted_sampler=True,
  )
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import stat
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    import torchvision.transforms as T

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    DataLoader = object
    Dataset = object
    WeightedRandomSampler = None
    T = None
    TORCH_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "01-data" / "Merged_Dataset" / "train"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_AUGMENTED_TRAIN_DIR = DEFAULT_SPLIT_DIR / "train_augmented"
DEFAULT_LABEL_MAP = PROJECT_ROOT / "04-features" / "label_map.json"
DEFAULT_CLASS_WEIGHTS = PROJECT_ROOT / "04-features" / "class_weights.npy"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = 224
DATASET_MEAN = (0.6320, 0.6092, 0.5805)
DATASET_STD = (0.2012, 0.1991, 0.2094)
RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.25


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
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
    data_dir: str | Path = DEFAULT_DATA_DIR,
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
    counts = Counter(labels)
    too_small = [label for label, count in counts.items() if count < 2]
    stratify = labels if not too_small else None
    if too_small:
        print(
            "[WARN] Some classes have fewer than 2 images; "
            "falling back to non-stratified split."
        )

    return train_test_split(
        paths,
        labels,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )


def make_transforms(
    image_size: int = TARGET_SIZE,
    mean: tuple[float, float, float] = DATASET_MEAN,
    std: tuple[float, float, float] = DATASET_STD,
    augment_train: bool = True,
) -> tuple[Callable, Callable]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch/torchvision is required to build image transforms.")

    train_steps: list[Callable] = [
        T.Resize((image_size, image_size)),
    ]
    if augment_train:
        train_steps.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=12),
                T.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.12,
                    hue=0.02,
                ),
            ]
        )
    train_steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    return T.Compose(train_steps), val_transform


class ImagePathDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str | Path],
        labels: Iterable[int],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use ImagePathDataset.")

        self.paths = [Path(p) for p in paths]
        self.labels = [int(y) for y in labels]
        self.transform = transform
        self.return_path = return_path

        if len(self.paths) != len(self.labels):
            raise ValueError("paths and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(path)
        return image, label


def compute_class_weights(labels: list[int], num_classes: int) -> np.ndarray:
    counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights


def make_weighted_sampler(labels: list[int], num_classes: int):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build a WeightedRandomSampler.")

    class_weights = compute_class_weights(labels, num_classes)
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
    return_path: bool = False,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    train_transform, val_transform = make_transforms(
        image_size=image_size,
        augment_train=augment_train,
    )

    train_dataset = ImagePathDataset(
        train_paths, train_labels, transform=train_transform, return_path=return_path
    )
    val_dataset = ImagePathDataset(
        val_paths, val_labels, transform=val_transform, return_path=return_path
    )

    num_classes = len(label_map)
    sampler = make_weighted_sampler(train_labels, num_classes) if use_weighted_sampler else None
    shuffle = sampler is None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    info = {
        "data_dir": str(Path(data_source)),
        "label_map": label_map,
        "idx_to_class": {idx: cls for cls, idx in label_map.items()},
        "num_classes": num_classes,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_counts": dict(sorted(Counter(train_labels).items())),
        "val_counts": dict(sorted(Counter(val_labels).items())),
        "class_weights": compute_class_weights(train_labels, num_classes),
        "image_size": image_size,
        "mean": DATASET_MEAN,
        "std": DATASET_STD,
    }
    return train_loader, val_loader, info


def build_dataloaders_from_split_folder(
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    batch_size: int = 32,
    image_size: int = TARGET_SIZE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
    return_path: bool = False,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("Install torch and torchvision to build dataloaders.")

    split_dir = Path(split_dir)
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected split folders: {train_dir} and {val_dir}"
        )

    label_map = load_label_map(label_map_path) or discover_label_map(train_dir)
    train_paths, train_labels, label_map = scan_image_folder(train_dir, label_map=label_map)
    val_paths, val_labels, _ = scan_image_folder(val_dir, label_map=label_map)

    return _make_dataloaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        label_map=label_map,
        data_source=split_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augment_train=augment_train,
        use_weighted_sampler=use_weighted_sampler,
        return_path=return_path,
        pin_memory=pin_memory,
    )


def build_dataloaders_from_train_val_dirs(
    train_dir: str | Path,
    val_dir: str | Path,
    batch_size: int = 32,
    image_size: int = TARGET_SIZE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
    return_path: bool = False,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("Install torch and torchvision to build dataloaders.")

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    label_map = load_label_map(label_map_path) or discover_label_map(train_dir)
    train_paths, train_labels, label_map = scan_image_folder(train_dir, label_map=label_map)
    val_paths, val_labels, _ = scan_image_folder(val_dir, label_map=label_map)

    return _make_dataloaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        label_map=label_map,
        data_source=train_dir.parent,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augment_train=augment_train,
        use_weighted_sampler=use_weighted_sampler,
        return_path=return_path,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    val_size: float = DEFAULT_VAL_SIZE,
    image_size: int = TARGET_SIZE,
    seed: int = RANDOM_STATE,
    num_workers: int = 0,
    augment_train: bool = True,
    use_weighted_sampler: bool = False,
    label_map_path: str | Path = DEFAULT_LABEL_MAP,
    return_path: bool = False,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    if not TORCH_AVAILABLE:
        raise ImportError("Install torch and torchvision to build dataloaders.")

    seed_everything(seed)
    data_dir = Path(data_dir)
    if (data_dir / "train").exists() and (data_dir / "val").exists():
        return build_dataloaders_from_split_folder(
            split_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            augment_train=augment_train,
            use_weighted_sampler=use_weighted_sampler,
            label_map_path=label_map_path,
            return_path=return_path,
            pin_memory=pin_memory,
        )

    paths, labels, label_map = scan_image_folder(data_dir, label_map_path=label_map_path)
    train_paths, val_paths, train_labels, val_labels = stratified_train_val_split(
        paths, labels, val_size=val_size, seed=seed
    )
    return _make_dataloaders_from_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        label_map=label_map,
        data_source=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augment_train=augment_train,
        use_weighted_sampler=use_weighted_sampler,
        return_path=return_path,
        pin_memory=pin_memory,
    )


def remove_tree(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return True

    def handle_remove_error(function, failed_path, _exc_info) -> None:
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    try:
        shutil.rmtree(path, onexc=handle_remove_error)
    except PermissionError as exc:
        print(f"[WARN] Could not remove {path}: {exc}")
        print("[WARN] Reusing existing split files where possible.")
        return False
    return True


def materialize_split_folder(
    train_paths: list[Path],
    val_paths: list[Path],
    train_labels: list[int],
    val_labels: list[int],
    label_map: dict[str, int],
    output_dir: str | Path = DEFAULT_SPLIT_DIR,
    overwrite: bool = False,
    link_mode: str = "copy",
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        existing_images = [
            p for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if existing_images and not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains images. "
                "Use overwrite=True or --overwrite-split to rebuild it."
            )
        if overwrite:
            remove_tree(output_dir)

    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    manifest_rows: list[dict[str, str | int]] = []

    def place_files(split: str, paths: list[Path], labels: list[int]) -> None:
        for src, label in zip(paths, labels):
            class_name = idx_to_class[int(label)]
            dst = output_dir / split / class_name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                manifest_rows.append(
                    {
                        "split": split,
                        "path": str(dst),
                        "source_path": str(src),
                        "label": int(label),
                        "class_name": class_name,
                    }
                )
                continue
            if link_mode == "hardlink":
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            elif link_mode == "copy":
                shutil.copy2(src, dst)
            else:
                raise ValueError("link_mode must be 'copy' or 'hardlink'.")
            manifest_rows.append(
                {
                    "split": split,
                    "path": str(dst),
                    "source_path": str(src),
                    "label": int(label),
                    "class_name": class_name,
                }
            )

    place_files("train", train_paths, train_labels)
    place_files("val", val_paths, val_labels)

    with open(output_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "path", "source_path", "label", "class_name"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def augment_split_train_folder(
    train_dir: str | Path,
    output_dir: str | Path = DEFAULT_AUGMENTED_TRAIN_DIR,
    target_count: int = 2500,
    label_map: dict[str, int] | None = None,
) -> None:
    train_dir = Path(train_dir)
    output_dir = Path(output_dir)
    classes = (
        [class_name for class_name, _ in sorted(label_map.items(), key=lambda item: item[1])]
        if label_map is not None
        else sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    )

    from preprocessing_pipeline import run_augmentation

    run_augmentation(
        src_dir=train_dir,
        dst_dir=output_dir,
        target_count=target_count,
        dry_run=False,
        classes=classes,
    )


def export_split_csv(
    output_path: str | Path,
    train_paths: list[Path],
    val_paths: list[Path],
    train_labels: list[int],
    val_labels: list[int],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "path", "label"])
        writer.writeheader()
        for split, paths, labels in (
            ("train", train_paths, train_labels),
            ("val", val_paths, val_labels),
        ):
            for path, label in zip(paths, labels):
                writer.writerow({"split": split, "path": str(path), "label": int(label)})


def summarize_counts(labels: list[int], idx_to_class: dict[int, str]) -> str:
    counts = Counter(labels)
    lines = []
    for idx in sorted(idx_to_class):
        lines.append(f"  {idx_to_class[idx]:<12}: {counts.get(idx, 0):>5}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val dataloaders.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--image-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--export-split", type=Path, default=None)
    parser.add_argument(
        "--augment-split-train",
        action="store_true",
        help="After splitting clean data, augment only split-dir/train using preprocessing_pipeline.",
    )
    parser.add_argument(
        "--augmented-train-dir",
        type=Path,
        default=None,
        help="Output folder for augmented train split.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=2500,
        help="Target images per class for --augment-split-train.",
    )
    parser.add_argument(
        "--make-split-folder",
        action="store_true",
        help="Create Prepared_Dataset/train and Prepared_Dataset/val folders.",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Rebuild split-dir if it already contains images.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use copy for independent files, or hardlink to save disk space.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_split_folder = (args.data_dir / "train").exists() and (args.data_dir / "val").exists()

    if is_split_folder:
        label_map = load_label_map(DEFAULT_LABEL_MAP) or discover_label_map(args.data_dir / "train")
        train_paths, train_labels, label_map = scan_image_folder(
            args.data_dir / "train", label_map=label_map
        )
        val_paths, val_labels, _ = scan_image_folder(
            args.data_dir / "val", label_map=label_map
        )
        paths = train_paths + val_paths
        labels = train_labels + val_labels
    else:
        paths, labels, label_map = scan_image_folder(args.data_dir)
        train_paths, val_paths, train_labels, val_labels = stratified_train_val_split(
            paths, labels, val_size=args.val_size, seed=args.seed
        )

    idx_to_class = {idx: cls for cls, idx in label_map.items()}

    print(f"Dataset     : {args.data_dir}")
    print(f"Split folder: {is_split_folder}")
    print(f"Images      : {len(paths)}")
    print(f"Classes     : {label_map}")
    print(f"Train/Val   : {len(train_paths)} / {len(val_paths)}")
    print("\nTrain counts:")
    print(summarize_counts(train_labels, idx_to_class))
    print("\nVal counts:")
    print(summarize_counts(val_labels, idx_to_class))

    if args.export_split is not None:
        export_split_csv(
            args.export_split,
            train_paths,
            val_paths,
            train_labels,
            val_labels,
        )
        print(f"\nSaved split CSV -> {args.export_split}")

    if args.make_split_folder:
        materialize_split_folder(
            train_paths=train_paths,
            val_paths=val_paths,
            train_labels=train_labels,
            val_labels=val_labels,
            label_map=label_map,
            output_dir=args.split_dir,
            overwrite=args.overwrite_split,
            link_mode=args.link_mode,
        )
        print(f"\nSaved split folder -> {args.split_dir}")
        print(f"Manifest          -> {args.split_dir / 'split_manifest.csv'}")

    if args.augment_split_train:
        split_train_dir = args.split_dir / "train"
        split_val_dir = args.split_dir / "val"
        if not split_train_dir.exists() or not split_val_dir.exists():
            materialize_split_folder(
                train_paths=train_paths,
                val_paths=val_paths,
                train_labels=train_labels,
                val_labels=val_labels,
                label_map=label_map,
                output_dir=args.split_dir,
                overwrite=args.overwrite_split,
                link_mode=args.link_mode,
            )
            print(f"\nSaved split folder -> {args.split_dir}")
            print(f"Manifest          -> {args.split_dir / 'split_manifest.csv'}")

        split_label_map = load_label_map(DEFAULT_LABEL_MAP) or discover_label_map(split_train_dir)
        augmented_train_dir = args.augmented_train_dir or (args.split_dir / "train_augmented")
        print("\n=== AUGMENT SPLIT TRAIN ONLY ===")
        augment_split_train_folder(
            train_dir=split_train_dir,
            output_dir=augmented_train_dir,
            target_count=args.target_per_class,
            label_map=split_label_map,
        )
        print(f"Augmented train   -> {augmented_train_dir}")
        print(f"Validation kept   -> {split_val_dir}")

    if TORCH_AVAILABLE:
        if args.augment_split_train:
            augmented_train_dir = args.augmented_train_dir or (args.split_dir / "train_augmented")
            train_loader, val_loader, info = build_dataloaders_from_train_val_dirs(
                train_dir=augmented_train_dir,
                val_dir=args.split_dir / "val",
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.num_workers,
                augment_train=False,
                use_weighted_sampler=args.weighted_sampler,
            )
        else:
            loader_data_dir = args.split_dir if args.make_split_folder else args.data_dir
            train_loader, val_loader, info = build_dataloaders(
                data_dir=loader_data_dir,
                batch_size=args.batch_size,
                val_size=args.val_size,
                image_size=args.image_size,
                seed=args.seed,
                num_workers=args.num_workers,
                augment_train=not args.no_augment,
                use_weighted_sampler=args.weighted_sampler,
            )
        images, targets = next(iter(train_loader))
        print(f"\nBatch image tensor: {tuple(images.shape)}")
        print(f"Batch label tensor: {tuple(targets.shape)}")
        print(f"Class weights     : {np.round(info['class_weights'], 4).tolist()}")
    else:
        print("\n[WARN] torch/torchvision is not installed; skipped DataLoader smoke test.")


if __name__ == "__main__":
    main()
