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
    RANDOM_STATE,
    augment_split_train_folder,
    discover_label_map,
    load_label_map,
    materialize_split_folder,
    scan_image_folder,
    stratified_train_val_split,
    summarize_counts,
)
from preprocessing_pipeline import BLUR_THRESHOLD, DEFAULT_CLEAN_SPLIT_DIR, clean_train_val_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command data pipeline: split train/val, clean each split, then augment train."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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


def main() -> None:
    args = parse_args()
    label_map = load_label_map(args.label_map)
    output_dir = args.output_dir
    temp_split_dir = output_dir / "_tmp_raw_split"

    print("=" * 60)
    print("STEP 1/3 - STRATIFIED TRAIN/VAL SPLIT")
    print("=" * 60)
    paths, labels, label_map = scan_image_folder(
        args.data_dir,
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
    print(f"Source     : {args.data_dir}")
    print(f"Output     : {output_dir}")
    print(f"Temp split : {temp_split_dir}")
    print(f"Images     : {len(paths)}")
    print(f"Train/Val  : {len(train_paths)} / {len(val_paths)}")
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
        output_dir=temp_split_dir,
        overwrite=args.overwrite_split,
        link_mode=args.link_mode,
    )
    print(f"\nSaved temp raw split -> {temp_split_dir}")

    print("\n" + "=" * 60)
    print("STEP 2/3 - CLEAN TRAIN/VAL SPLITS")
    print("=" * 60)
    clean_args = SimpleNamespace(
        split_dir=temp_split_dir,
        clean_split_output=output_dir,
        dry_run=args.dry_run_clean,
        remove_blur=args.remove_blur,
        remove_near_dup=args.remove_near_dup,
        blur_threshold=args.blur_threshold,
        phash_threshold=args.phash_threshold,
        phash_sample=args.phash_sample,
    )
    clean_train_val_splits(clean_args)
    print(f"\nClean split -> {output_dir}")

    if args.skip_augment:
        print("\nSTEP 3/3 - AUGMENT TRAIN skipped.")
        if not args.keep_temp_split and temp_split_dir.exists() and not args.dry_run_clean:
            shutil.rmtree(temp_split_dir)
            print(f"Removed temp split -> {temp_split_dir}")
        return

    print("\n" + "=" * 60)
    print("STEP 3/3 - AUGMENT CLEAN TRAIN ONLY")
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

    if not args.keep_temp_split and temp_split_dir.exists() and not args.dry_run_clean:
        shutil.rmtree(temp_split_dir)
        print(f"Removed temp split -> {temp_split_dir}")


if __name__ == "__main__":
    main()
