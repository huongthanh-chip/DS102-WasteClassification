#!/usr/bin/env python3
"""
Preprocessing Pipeline — Waste Classification Dataset
=====================================================
Produces:
  1. Cleaned_Dataset/train/   : images after dedup, low-content & blur removal
  2. Augmented_Dataset/train/ : class-balanced augmented images
  3. Features/                : HOG+Color features, class weights, label map

Usage:
  python preprocessing_pipeline.py --clean --augment --features
  python preprocessing_pipeline.py --clean --augment --dry-run
  python preprocessing_pipeline.py --augment --target-per-class 2000

Recheck & fixes vs. the original notebook:
  * MD5 uses chunk-based hashing (safer for large files).
  * Near-duplicate (pHash) results are now actually excluded from Cleaned_Dataset.
  * `count_images_by_class` and `run_augmentation` filter by IMAGE_EXTS.
  * Albumentations 2.x API fixed: `GaussNoise(var_limit=...)` -> `GaussNoise(std_range=...)`.
  * Albumentations transforms are wrapped so `aug_strong / aug_light` are always
    callables: `Image.Image -> Image.Image`.
  * `build_cleaned_dataset` correctly handles both `dup_groups` and `near_dup_pairs`.
  * Added `--phash-sample` to quickly test pHash on a subset.
"""

import argparse
import hashlib
import json
import random
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog
from tqdm import tqdm

# -------------------------------------------------------------
# Optional libraries
# -------------------------------------------------------------
try:
    import albumentations as A
    ALBU_AVAILABLE = True
except Exception as exc:
    A = None
    ALBU_AVAILABLE = False
    warnings.warn(f"Albumentations not available: {exc}. Augmentation will use PIL fallback.")

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception as exc:
    TORCH_AVAILABLE = False
    warnings.warn(f"PyTorch not available: {exc}. CNN feature extraction disabled.")

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
random.seed(42)
np.random.seed(42)

DATA_DIR   = Path("train")
CLEAN_DIR  = Path("Cleaned_Dataset/train")
AUG_DIR    = Path("Augmented_Dataset/train")
FEAT_DIR   = Path("Features")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLASSES    = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()]) if DATA_DIR.exists() else []

DATASET_MEAN = (0.6320, 0.6092, 0.5805)
DATASET_STD  = (0.2012, 0.1991, 0.2094)
TARGET_SIZE  = 224

LOW_CONTENT_THRESHOLD = 10.0
BLUR_THRESHOLD        = 20.0

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def get_images(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        return []
    pattern = root.rglob("*") if recursive else root.iterdir()
    return [p for p in pattern if p.suffix.lower() in IMAGE_EXTS]


def count_images_by_class(root: Path) -> Dict[str, int]:
    counts = {}
    for cls in CLASSES:
        cls_dir = root / cls
        counts[cls] = len(get_images(cls_dir)) if cls_dir.exists() else 0
    return counts


def compute_md5(filepath: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------
# 1. EXACT DEDUPLICATION (MD5)
# -------------------------------------------------------------
def exact_dedup(data_dir: Path, verbose: bool = True) -> Dict[str, List[Path]]:
    all_imgs = get_images(data_dir)
    hash_map = defaultdict(list)
    for path in tqdm(all_imgs, desc="MD5 hashing"):
        try:
            hash_map[compute_md5(path)].append(path)
        except Exception as e:
            print(f"  [WARN] MD5 failed for {path.name}: {e}")

    dup_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    if verbose:
        n_dup = sum(len(v) - 1 for v in dup_groups.values())
        print(f"Exact dup groups : {len(dup_groups)}")
        print(f"Files to remove  : {n_dup}")
        print(f"Ratio            : {n_dup / max(len(all_imgs), 1) * 100:.1f}%")
    return dup_groups


def resolve_paths_to_remove(dup_groups: Dict[str, List[Path]]) -> Set[Path]:
    """Keep ds1_ first, then sort by name; remove the rest."""
    to_remove = set()
    for paths in dup_groups.values():
        paths_sorted = sorted(
            paths, key=lambda p: (0 if "ds1_" in p.name else 1, p.name)
        )
        for p in paths_sorted[1:]:
            to_remove.add(p.resolve())
    return to_remove


# -------------------------------------------------------------
# 1.2 NEAR-DUPLICATE (pHash)
# -------------------------------------------------------------
def compute_phash(img_path: Path, hash_size: int = 8) -> int:
    img = Image.open(img_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    arr = np.array(img)
    diff = arr[:, 1:] > arr[:, :-1]
    return sum(2 ** i for i, v in enumerate(diff.flatten()) if v)


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def find_near_dups(
    data_dir: Path,
    threshold: int = 5,
    sample_per_class: int | None = None,
) -> List[Tuple[Path, Path, int]]:
    all_imgs = get_images(data_dir)

    if sample_per_class:
        sampled = []
        for cls in CLASSES:
            cls_imgs = [p for p in all_imgs if p.parent.name == cls]
            if cls_imgs:
                sampled.extend(
                    random.sample(cls_imgs, min(sample_per_class, len(cls_imgs)))
                )
        all_imgs = sampled

    print(f"Computing pHash for {len(all_imgs)} images...")
    hashes = {}
    for path in tqdm(all_imgs, desc="pHash"):
        try:
            hashes[path] = compute_phash(path)
        except Exception:
            pass

    # Pairwise within each class (O(n^2) per class)
    near_dup_pairs = []
    class_groups = defaultdict(list)
    for p in hashes.keys():
        class_groups[p.parent.name].append(p)

    for cls, cls_paths in class_groups.items():
        for i in range(len(cls_paths)):
            for j in range(i + 1, len(cls_paths)):
                d = hamming_distance(hashes[cls_paths[i]], hashes[cls_paths[j]])
                if d <= threshold:
                    near_dup_pairs.append((cls_paths[i], cls_paths[j], d))

    print(f"Near-duplicate pairs (threshold={threshold}): {len(near_dup_pairs)}")
    return near_dup_pairs


def resolve_near_dup_paths_to_remove(
    near_dup_pairs: List[Tuple[Path, Path, int]]
) -> Set[Path]:
    """
    Build connected components from near-duplicate pairs.
    In each component, keep one image (prefer ds1_) and remove the rest.
    """
    # Union-Find
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for p1, p2, _ in near_dup_pairs:
        union(p1.resolve(), p2.resolve())

    # Group by component root
    comps = defaultdict(list)
    for p in parent:
        comps[find(p)].append(p)

    to_remove = set()
    for comp_paths in comps.values():
        sorted_paths = sorted(
            comp_paths, key=lambda p: (0 if "ds1_" in p.name else 1, p.name)
        )
        for p in sorted_paths[1:]:
            to_remove.add(p)
    return to_remove


# -------------------------------------------------------------
# 1.3 LOW-CONTENT FILTER
# -------------------------------------------------------------
def find_low_content_images(
    data_dir: Path, threshold: float = LOW_CONTENT_THRESHOLD
) -> pd.DataFrame:
    rows = []
    for path in tqdm(get_images(data_dir), desc="Low-content scan"):
        try:
            gray = np.array(Image.open(path).convert("L"))
            std = gray.std()
            if std < threshold:
                rows.append(
                    {"path": str(path), "class": path.parent.name, "std": round(std, 2)}
                )
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Low-content images (std < {threshold}): {len(df)}")
        print(df.groupby("class")["std"].describe().round(2))
    else:
        print(f"No low-content images found (threshold={threshold}).")
    return df


# -------------------------------------------------------------
# 1.4 BLUR DETECTION
# -------------------------------------------------------------
def compute_laplacian_var(img_path: Path) -> float:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def find_blurry_images(
    data_dir: Path, threshold: float = BLUR_THRESHOLD
) -> pd.DataFrame:
    rows = []
    for path in tqdm(get_images(data_dir), desc="Blur detection"):
        lap_var = compute_laplacian_var(path)
        rows.append(
            {
                "path": str(path),
                "class": path.parent.name,
                "lap_var": lap_var,
                "is_blurry": (lap_var < threshold) and (lap_var >= 0),
            }
        )
    df = pd.DataFrame(rows)
    print(f"Blurry images (Laplacian var < {threshold}): {df['is_blurry'].sum()}")
    print(df.groupby("class")["is_blurry"].sum().to_string())
    return df


# -------------------------------------------------------------
# 1.5 BUILD CLEANED DATASET
# -------------------------------------------------------------
def build_cleaned_dataset(
    src_dir: Path,
    dst_dir: Path,
    dup_groups: Dict[str, List[Path]],
    near_dup_pairs: List[Tuple[Path, Path, int]],
    df_low: pd.DataFrame,
    df_blur: pd.DataFrame,
    remove_blur: bool = True,
    dry_run: bool = True,
) -> List[Path]:
    exclude: Set[Path] = set()

    # Exact dups
    exclude.update(resolve_paths_to_remove(dup_groups))

    # Near dups
    if near_dup_pairs:
        exclude.update(resolve_near_dup_paths_to_remove(near_dup_pairs))

    # Low-content
    if not df_low.empty:
        for p in df_low["path"]:
            exclude.add(Path(p).resolve())

    # Blurry
    if remove_blur and not df_blur.empty:
        for _, row in df_blur[df_blur["is_blurry"]].iterrows():
            exclude.add(Path(row["path"]).resolve())

    all_imgs = get_images(src_dir)
    keep = [p for p in all_imgs if p.resolve() not in exclude]

    print(f"Total images to exclude: {len(exclude)}")
    print(f"Images kept             : {len(keep)} / {len(all_imgs)}")

    if not dry_run:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        for src_path in tqdm(keep, desc="Copy cleaned dataset"):
            rel = src_path.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)

        print("\n=== Stats after cleaning ===")
        for cls in CLASSES:
            n = len(get_images(dst_dir / cls))
            print(f"  {cls:<15}: {n}")

    return keep


# -------------------------------------------------------------
# 2. IMAGE TRANSFORMATION
# -------------------------------------------------------------
def preprocess_image(
    img_path: Path,
    size: int = TARGET_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if normalize:
        mean = np.array(DATASET_MEAN, dtype=np.float32)
        std = np.array(DATASET_STD, dtype=np.float32)
        arr = (arr - mean) / (std + 1e-7)
    return arr


# -------------------------------------------------------------
# 3. FEATURE EXTRACTION
# -------------------------------------------------------------
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    channel_axis=-1,
    transform_sqrt=True,
    feature_vector=True,
)


def extract_hog(img_path: Path, size: int = TARGET_SIZE) -> np.ndarray:
    arr = (preprocess_image(img_path, size=size, normalize=False) * 255).astype(np.uint8)
    return hog(arr, **HOG_PARAMS)


N_BINS = 32


def extract_color_hist(
    img_path: Path, n_bins: int = N_BINS, normalize: bool = True
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    arr = np.array(img)
    hists = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=n_bins, range=(0, 256))
        if normalize:
            hist = hist / (hist.sum() + 1e-7)
        hists.append(hist)
    return np.concatenate(hists).astype(np.float32)


# CNN embedding (optional)
if TORCH_AVAILABLE:
    _cnn_model = models.resnet50(weights="IMAGENET1K_V2")
    _cnn_model.fc = torch.nn.Identity()
    _cnn_model.eval()

    _cnn_transform = T.Compose(
        [
            T.Resize((TARGET_SIZE, TARGET_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )

    def extract_cnn_embedding(img_path: Path) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        tensor = _cnn_transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = _cnn_model(tensor).squeeze().numpy()
        return emb.astype(np.float32)
else:
    extract_cnn_embedding = None  # type: ignore


# -------------------------------------------------------------
# 3.4 BUILD FEATURE MATRIX
# -------------------------------------------------------------
def build_feature_matrix(
    data_dir: Path,
    mode: str = "hog+color",
    fallback_dir: Path | None = None,
):
    if "cnn" in mode and not TORCH_AVAILABLE:
        warnings.warn("Torch unavailable — CNN features will be skipped.")

    X_list, y_list = [], []
    le = LabelEncoder()

    all_imgs = []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            all_imgs.extend([(p, cls) for p in get_images(cls_dir)])

    if not all_imgs and fallback_dir is not None and fallback_dir != data_dir:
        print(f"[WARN] No images in {data_dir}, fallback to {fallback_dir}")
        for cls in CLASSES:
            cls_dir = fallback_dir / cls
            if cls_dir.exists():
                all_imgs.extend([(p, cls) for p in get_images(cls_dir)])
        data_dir = fallback_dir

    if not all_imgs:
        raise FileNotFoundError(f"No images found in {data_dir}.")

    for img_path, cls in tqdm(all_imgs, desc=f"Extracting [{mode}]"):
        feats = []
        try:
            if "hog" in mode:
                feats.append(extract_hog(img_path))
            if "color" in mode:
                feats.append(extract_color_hist(img_path))
            if "cnn" in mode and TORCH_AVAILABLE:
                feats.append(extract_cnn_embedding(img_path))
            if feats:
                X_list.append(np.concatenate(feats))
                y_list.append(cls)
        except Exception as e:
            print(f"  [SKIP] {img_path.name}: {e}")

    if not X_list:
        raise ValueError("No features extracted.")

    X = np.array(X_list, dtype=np.float32)
    y = le.fit_transform(y_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix: {X.shape}")
    print(f"Classes: {list(zip(le.classes_, range(len(le.classes_))))}")

    return X_scaled, y, le, scaler


# -------------------------------------------------------------
# 4. DATA AUGMENTATION
# -------------------------------------------------------------
def pil_random_resized_crop(
    img: Image.Image,
    scale=(0.7, 1.0),
    ratio=(0.75, 1.33),
    size: int = TARGET_SIZE,
) -> Image.Image:
    w, h = img.size
    area = w * h
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(*ratio)
        new_w = int(round((target_area * aspect) ** 0.5))
        new_h = int(round((target_area / aspect) ** 0.5))
        if 0 < new_w <= w and 0 < new_h <= h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            return img.resize((size, size), Image.LANCZOS)
    # Fallback to center crop
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    return img.resize((size, size), Image.LANCZOS)


def pil_add_noise(img: Image.Image, sigma_range=(10, 25)) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_augment(img: Image.Image, strength: str = "strong") -> Image.Image:
    if strength == "strong":
        img = pil_random_resized_crop(img, scale=(0.7, 1.0), ratio=(0.75, 1.33))
    else:
        img = pil_random_resized_crop(img, scale=(0.85, 1.0), ratio=(0.9, 1.1))

    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    if strength == "strong":
        if random.random() < 0.7:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-0.3, 0.3))
        if random.random() < 0.7:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-0.3, 0.3))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(1 + random.uniform(-0.3, 0.3))
        if random.random() < 0.5:
            img = img.rotate(
                random.uniform(-30, 30), resample=Image.BILINEAR, expand=True
            )
        if random.random() < 0.3:
            img = pil_add_noise(img)
    else:
        if random.random() < 0.4:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-0.15, 0.15))
        if random.random() < 0.4:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-0.15, 0.15))
        if random.random() < 0.3:
            img = ImageEnhance.Color(img).enhance(1 + random.uniform(-0.15, 0.15))
        if random.random() < 0.3:
            img = img.rotate(
                random.uniform(-10, 10), resample=Image.BILINEAR, expand=True
            )

    img = ImageOps.fit(
        img, (TARGET_SIZE, TARGET_SIZE), method=Image.LANCZOS, centering=(0.5, 0.5)
    )
    return img


# Build augmentation callables that always take / return PIL Image
if ALBU_AVAILABLE and A is not None:
    _alb_aug_strong = A.Compose(
        [
            A.RandomResizedCrop(size=(TARGET_SIZE, TARGET_SIZE), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # Albumentations 2.x uses std_range instead of var_limit for GaussNoise
            A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.2,
            ),
            A.Resize(TARGET_SIZE, TARGET_SIZE),
        ]
    )

    _alb_aug_light = A.Compose(
        [
            A.RandomResizedCrop(size=(TARGET_SIZE, TARGET_SIZE), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Resize(TARGET_SIZE, TARGET_SIZE),
        ]
    )

    def _make_pil_wrapper(alb_pipeline: A.Compose) -> Callable[[Image.Image], Image.Image]:
        def _wrap(img: Image.Image) -> Image.Image:
            arr = np.array(img.convert("RGB"))
            result = alb_pipeline(image=arr)
            return Image.fromarray(result["image"])
        return _wrap

    aug_strong: Callable[[Image.Image], Image.Image] = _make_pil_wrapper(_alb_aug_strong)
    aug_light: Callable[[Image.Image], Image.Image] = _make_pil_wrapper(_alb_aug_light)
else:
    aug_strong = lambda img: pil_augment(img, "strong")  # type: ignore
    aug_light = lambda img: pil_augment(img, "light")    # type: ignore


# -------------------------------------------------------------
def augment_class(
    cls_dir: Path,
    dst_dir: Path,
    target_count: int,
    aug_pipeline: Callable[[Image.Image], Image.Image],
    dry_run: bool = True,
) -> int:
    cls_name = cls_dir.name
    src_imgs = get_images(cls_dir)
    n_current = len(src_imgs)
    n_needed = max(0, target_count - n_current)

    print(f"  {cls_name:<15}: {n_current} images -> need {n_needed} more")

    if n_current == 0:
        if dry_run:
            return n_needed
        raise FileNotFoundError(f"No source images in {cls_dir}.")

    dst_cls = dst_dir / cls_name
    if not dry_run:
        dst_cls.mkdir(parents=True, exist_ok=True)
        # Copy originals
        for p in src_imgs:
            shutil.copy2(p, dst_cls / p.name)

    if dry_run or n_needed == 0:
        return n_needed

    aug_count = 0
    while aug_count < n_needed:
        src = random.choice(src_imgs)
        aug_img = aug_pipeline(Image.open(src).convert("RGB"))
        out_name = f"aug_{aug_count:05d}_{src.name}"
        aug_img.save(dst_cls / out_name)
        aug_count += 1

    print(f"    -> Created {aug_count} augmented images")
    return aug_count


def run_augmentation(
    src_dir: Path,
    dst_dir: Path,
    target_count: int = 2500,
    dry_run: bool = True,
):
    if dst_dir.exists() and not dry_run:
        shutil.rmtree(dst_dir)

    for cls in CLASSES:
        cls_dir = src_dir / cls
        n_imgs = len(get_images(cls_dir))
        pipeline = aug_strong if n_imgs < 1500 else aug_light
        augment_class(cls_dir, dst_dir, target_count, pipeline, dry_run=dry_run)

    if not dry_run:
        print("\n=== Stats after augmentation ===")
        for cls in CLASSES:
            n = len(get_images(dst_dir / cls))
            print(f"  {cls:<15}: {n}")


# -------------------------------------------------------------
# 4.3 CLASS WEIGHTS
# -------------------------------------------------------------
def save_class_weights(root: Path):
    counts = count_images_by_class(root)
    if sum(counts.values()) == 0:
        raise ValueError(f"No images found in {root} for class-weight computation.")

    labels = []
    for cls_idx, cls in enumerate(CLASSES):
        labels.extend([cls_idx] * counts[cls])
    labels = np.array(labels, dtype=int)

    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASSES)), y=labels
    )
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEAT_DIR / "class_weights.npy", weights)
    with open(FEAT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(dict(zip(CLASSES, range(len(CLASSES)))), f, indent=2, ensure_ascii=False)

    print("\n=== Class weights (balanced) ===")
    for cls, w in zip(CLASSES, weights):
        bar = "#" * int(w * 5)
        print(f"  {cls:<15}: {w:.4f}  {bar}")
    print(f"Saved -> {FEAT_DIR}/class_weights.npy , label_map.json")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Waste Classification Preprocessing Pipeline"
    )
    p.add_argument("--clean", action="store_true", help="Build Cleaned_Dataset")
    p.add_argument("--augment", action="store_true", help="Build Augmented_Dataset")
    p.add_argument("--features", action="store_true", help="Extract HOG+Color features")
    p.add_argument("--dry-run", action="store_true", help="Only print stats, do NOT write files")
    p.add_argument("--remove-blur", action=argparse.BooleanOptionalAction, default=True, help="Remove blurry images (default: on)")
    p.add_argument("--remove-near-dup", action=argparse.BooleanOptionalAction, default=False, help="Run pHash near-duplicate removal (slow, default: off)")
    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    p.add_argument("--phash-threshold", type=int, default=5)
    p.add_argument("--phash-sample", type=int, default=None, help="Sample N images per class for pHash (fast test)")
    p.add_argument("--target-per-class", type=int, default=2500, help="Target count per class after augmentation")
    p.add_argument("--feature-mode", type=str, default="hog+color", choices=["hog+color", "cnn", "hog+color+cnn"])
    return p.parse_args()


def main():
    args = parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if not CLASSES:
        raise ValueError(f"No class subdirectories found in {DATA_DIR}")

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Dataset path : {DATA_DIR}")
    print(f"Classes      : {CLASSES}")
    print(f"Dry run      : {args.dry_run}")
    print()

    dup_groups = {}
    near_dup_pairs = []
    df_low = pd.DataFrame()
    df_blur = pd.DataFrame()

    # -- Step 1: Cleaning analysis --
    print("=== EXACT DEDUPLICATION (MD5) ===")
    dup_groups = exact_dedup(DATA_DIR)

    if args.remove_near_dup:
        print("\n=== NEAR-DUPLICATE (pHash) ===")
        near_dup_pairs = find_near_dups(
            DATA_DIR,
            threshold=args.phash_threshold,
            sample_per_class=args.phash_sample,
        )
    else:
        print("\n=== NEAR-DUPLICATE (pHash) ===")
        print("Skipped (--remove-near-dup not set).")

    print("\n=== LOW-CONTENT FILTER ===")
    df_low = find_low_content_images(DATA_DIR)

    print("\n=== BLUR DETECTION ===")
    df_blur = find_blurry_images(DATA_DIR, threshold=args.blur_threshold)

    # -- Step 1.5: Build cleaned dataset --
    if args.clean or args.augment or args.features:
        print("\n=== BUILD CLEANED DATASET ===")
        kept = build_cleaned_dataset(
            src_dir=DATA_DIR,
            dst_dir=CLEAN_DIR,
            dup_groups=dup_groups,
            near_dup_pairs=near_dup_pairs,
            df_low=df_low,
            df_blur=df_blur,
            remove_blur=args.remove_blur,
            dry_run=args.dry_run,
        )
    else:
        kept = []

    # -- Step 3: Features (from Cleaned_Dataset or fallback DATA_DIR) --
    if args.features:
        print("\n=== FEATURE EXTRACTION ===")
        feat_src = CLEAN_DIR if (not args.dry_run and CLEAN_DIR.exists()) else DATA_DIR
        X, y, le, scaler = build_feature_matrix(
            feat_src, mode=args.feature_mode, fallback_dir=DATA_DIR
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        FEAT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            FEAT_DIR / f"features_{args.feature_mode}.npz",
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
        )
        print(f"Saved -> {FEAT_DIR}/features_{args.feature_mode}.npz")

    # -- Step 4: Augmentation --
    if args.augment:
        print("\n=== AUGMENTATION PLAN ===")
        plan_root = CLEAN_DIR if (not args.dry_run and CLEAN_DIR.exists()) else DATA_DIR
        counts = count_images_by_class(plan_root)
        print(f"Target per class: {args.target_per_class}\n")
        for cls in CLASSES:
            n = counts.get(cls, 0)
            need = max(0, args.target_per_class - n)
            strategy = (
                "STRONG aug" if n < 1500 else ("LIGHT aug" if n < args.target_per_class else "No aug needed")
            )
            print(f"  {cls:<15}: {n:>5} images | need {need:>4} | {strategy}")

        print("\n=== RUN AUGMENTATION ===")
        aug_src = CLEAN_DIR if (not args.dry_run and CLEAN_DIR.exists()) else DATA_DIR
        run_augmentation(
            aug_src, AUG_DIR, target_count=args.target_per_class, dry_run=args.dry_run
        )

    # -- Class weights (from cleaned or augmented if available) --
    if (args.clean or args.augment) and not args.dry_run:
        weight_root = AUG_DIR if (args.augment and AUG_DIR.exists()) else CLEAN_DIR
        if weight_root.exists() and any(get_images(weight_root / cls) for cls in CLASSES):
            print(f"\n=== CLASS WEIGHTS ===")
            save_class_weights(weight_root)

    # -- Summary --
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_original = sum(len(get_images(DATA_DIR / cls)) for cls in CLASSES)
    n_cleaned = (
        sum(len(get_images(CLEAN_DIR / cls)) for cls in CLASSES)
        if CLEAN_DIR.exists() else "(not created)"
    )
    n_aug = (
        sum(len(get_images(AUG_DIR / cls)) for cls in CLASSES)
        if AUG_DIR.exists() else "(not created)"
    )
    print(f"Original images : {n_original:,}")
    print(f"Cleaned images  : {n_cleaned}")
    print(f"Augmented images: {n_aug}")
    print("Done.")


if __name__ == "__main__":
    main()
