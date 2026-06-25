#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataloader import build_dataloaders_from_train_val_dirs
from train_cnn import create_efficientnet_b0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "05-models" / "efficientnet_b0" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "05-models" / "efficientnet_b0" / "test_metrics.json"


def resolve_test_dir(split_dir: Path) -> Path:
    direct = split_dir / "test"
    nested = direct / "test"
    if nested.exists():
        return nested
    return direct


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets_all = []
    preds_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        preds_all.extend(preds)
        targets_all.extend(targets.cpu().tolist())
    return targets_all, preds_all


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B0 on test split.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]
    idx_to_class = {int(idx): cls for idx, cls in checkpoint["idx_to_class"].items()}
    target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    test_dir = args.test_dir or resolve_test_dir(args.split_dir)
    # Reuse the train/val loader helper by passing test_dir as val_dir.
    _, test_loader, _ = build_dataloaders_from_train_val_dirs(
        train_dir=test_dir,
        val_dir=test_dir,
        batch_size=args.batch_size,
        image_size=int(checkpoint.get("image_size", 224)),
        num_workers=args.num_workers,
        augment_train=False,
        use_weighted_sampler=False,
        label_map_path=checkpoint["args"]["label_map"],
    )

    model = create_efficientnet_b0(
        num_classes=len(label_map),
        pretrained=False,
        dropout=float(checkpoint["args"].get("dropout", 0.4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    y_true, y_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"checkpoint: {args.checkpoint}")
    print(f"test_dir  : {test_dir}")
    print(f"accuracy  : {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))
    print("confusion_matrix:")
    print(np.asarray(cm))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "test_dir": str(test_dir),
                "accuracy": acc,
                "classification_report": report,
                "confusion_matrix": cm,
            },
            f,
            indent=2,
        )
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
