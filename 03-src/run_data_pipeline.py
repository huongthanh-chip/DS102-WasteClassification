#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

from dataloader import (
    DEFAULT_DATA_DIR,
    DEFAULT_LABEL_MAP,
    DEFAULT_VAL_SIZE,
    IMAGE_EXTS,
    RANDOM_STATE,
    augment_split_train_folder,
    discover_label_map,
    load_label_map,
    materialize_split_folder,
    scan_image_folder,
    stratified_train_val_split,
    summarize_counts,
)
from preprocessing_pipeline import (
    BLUR_THRESHOLD,
    DEFAULT_CLEAN_SPLIT_DIR,
    clean_one_split,
    set_classes_from,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command data pipeline: clean merged train pool, split train/val, optionally clean test, then augment train."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CLEAN_SPLIT_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--target-per-class", type=int, default=2500)
    parser.add_argument("--link-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--keep-temp-split", action="store_true")
    parser.add_argument("--dry-run-clean", action="store_true")
    parser.add_argument("--remove-blur", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    parser.add_argument("--remove-near-dup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phash-threshold", type=int, default=5)
    parser.add_argument("--phash-sample", type=int, default=None)
    parser.add_argument("--skip-augment", action="store_true")
    return parser.parse_args()


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clean_args_for(src_dir: Path, dst_dir: Path, args: argparse.Namespace, light: bool = False):
    return SimpleNamespace(
        src_dir=src_dir,
        dst_dir=dst_dir,
        dry_run=args.dry_run_clean,
        remove_blur=False if light else args.remove_blur,
        remove_near_dup=False if light else args.remove_near_dup,
        blur_threshold=args.blur_threshold,
        phash_threshold=args.phash_threshold,
        phash_sample=args.phash_sample,
    )


def resolve_class_folder(path: Path, label_map: dict[str, int]) -> Path:
    if any((path / class_name).is_dir() for class_name in label_map):
        return path
    nested = path / path.name
    if nested.exists() and any((nested / class_name).is_dir() for class_name in label_map):
        return nested
    return path


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    args = parse_args()
    label_map = load_label_map(args.label_map)
    output_dir = args.output_dir
    temp_root = output_dir.parent / f"_{output_dir.name}_tmp"
    temp_clean_pool = temp_root / "clean_pool"
    test_backup_dir = output_dir.parent / f"_{output_dir.name}_test_backup"

    if args.overwrite_split:
        if args.test_dir is not None:
            test_dir_resolved = args.test_dir.resolve(strict=False)
            output_dir_resolved = output_dir.resolve(strict=False)
            if test_dir_resolved == output_dir_resolved or output_dir_resolved in test_dir_resolved.parents:
                remove_dir(test_backup_dir)
                if args.test_dir.exists():
                    shutil.copytree(args.test_dir, test_backup_dir)
                    args.test_dir = test_backup_dir
                    print(f"Backed up test source before overwrite -> {test_backup_dir}")
        remove_dir(temp_root)
        remove_dir(output_dir)

    print("=" * 60)
    print("STEP 1/4 - CLEAN MERGED TRAIN POOL")
    print("=" * 60)
    set_classes_from(args.data_dir)
    pool_clean_args = clean_args_for(args.data_dir, temp_clean_pool, args)
    kept_pool = clean_one_split(
        src_dir=args.data_dir,
        dst_dir=temp_clean_pool,
        args=pool_clean_args,
    )
    if args.dry_run_clean:
        print("\nDry-run clean finished. Stop before split because no cleaned files were written.")
        return

    print("\n" + "=" * 60)
    print("STEP 2/4 - STRATIFIED TRAIN/VAL SPLIT FROM CLEANED POOL")
    print("=" * 60)
    paths, labels, label_map = scan_image_folder(
        temp_clean_pool,
        label_map=label_map,
        label_map_path=args.label_map,
    )
    train_paths, val_paths, train_labels, val_labels = stratified_train_val_split(
        paths,
        labels,
        val_size=args.val_size,
        seed=args.seed,
    )
    idx_to_class = {idx: cls for cls, idx in label_map.items()}
    print(f"Raw source     : {args.data_dir}")
    print(f"Cleaned pool   : {temp_clean_pool}")
    print(f"Output prepared: {output_dir}")
    print(f"Images cleaned : {len(paths)}")
    print(f"Train/Val      : {len(train_paths)} / {len(val_paths)}")
    print("\nTrain counts:")
    print(summarize_counts(train_labels, idx_to_class))
    print("\nVal counts:")
    print(summarize_counts(val_labels, idx_to_class))

    materialize_split_folder(
        train_paths=train_paths,
        val_paths=val_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        label_map=label_map,
        output_dir=output_dir,
        overwrite=False,
        link_mode=args.link_mode,
    )
    print(f"\nSaved prepared train/val -> {output_dir}")

    if args.test_dir is not None:
        print("\n" + "=" * 60)
        print("STEP 3/4 - LIGHT CLEAN TEST SPLIT")
        print("=" * 60)
        test_dir = resolve_class_folder(args.test_dir, label_map)
        print(f"Test source: {test_dir}")
        n_test_images = count_images(test_dir)
        if n_test_images == 0:
            raise ValueError(
                f"No test images found in {test_dir}. "
                "Check --test-dir and quote paths that contain '-' or spaces."
            )
        print(f"Test images: {n_test_images}")
        test_clean_args = clean_args_for(test_dir, output_dir / "test", args, light=True)
        clean_one_split(
            src_dir=test_dir,
            dst_dir=output_dir / "test",
            args=test_clean_args,
        )
    else:
        print("\nSTEP 3/4 - LIGHT CLEAN TEST SPLIT skipped (--test-dir not provided).")

    if args.skip_augment:
        print("\nSTEP 4/4 - AUGMENT TRAIN skipped.")
        if not args.keep_temp_split and temp_root.exists():
            shutil.rmtree(temp_root)
            print(f"Removed temp files -> {temp_root}")
        if not args.keep_temp_split and test_backup_dir.exists():
            shutil.rmtree(test_backup_dir)
            print(f"Removed test backup -> {test_backup_dir}")
        return

    print("\n" + "=" * 60)
    print("STEP 4/4 - AUGMENT PREPARED TRAIN ONLY")
    print("=" * 60)
    split_label_map = load_label_map(args.label_map) or discover_label_map(output_dir / "train")
    augmented_train_dir = output_dir / "train_augmented"
    augment_split_train_folder(
        train_dir=output_dir / "train",
        output_dir=augmented_train_dir,
        target_count=args.target_per_class,
        label_map=split_label_map,
    )
    print(f"\nAugmented train -> {augmented_train_dir}")
    print(f"Validation kept -> {output_dir / 'val'}")

    if not args.keep_temp_split and temp_root.exists():
        shutil.rmtree(temp_root)
        print(f"Removed temp files -> {temp_root}")
    if not args.keep_temp_split and test_backup_dir.exists():
        shutil.rmtree(test_backup_dir)
        print(f"Removed test backup -> {test_backup_dir}")


if __name__ == "__main__":
    main()
