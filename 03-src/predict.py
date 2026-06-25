#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from dataloader import IMAGE_EXTS, TARGET_SIZE, DATASET_MEAN, DATASET_STD
from train_cnn import create_efficientnet_b0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "05-models" / "efficientnet_b0" / "best.pt"


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_map = checkpoint["label_map"]
    idx_to_class = {int(idx): cls for idx, cls in checkpoint["idx_to_class"].items()}
    model = create_efficientnet_b0(num_classes=len(label_map), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_class, checkpoint


def make_predict_transform(checkpoint: dict):
    image_size = int(checkpoint.get("image_size", TARGET_SIZE))
    mean = tuple(checkpoint.get("mean", DATASET_MEAN))
    std = tuple(checkpoint.get("std", DATASET_STD))
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


@torch.no_grad()
def predict_paths(model, paths: list[Path], transform, idx_to_class, device, topk: int):
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, probs.numel())
        values, indices = torch.topk(probs, k=k)
        row = {
            "path": str(path),
            "pred_label": int(indices[0].item()),
            "pred_class": idx_to_class[int(indices[0].item())],
            "confidence": float(values[0].item()),
        }
        for rank, (prob, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            row[f"top{rank}_class"] = idx_to_class[int(idx)]
            row[f"top{rank}_prob"] = float(prob)
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Predict waste class with EfficientNet-B0 checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or folder.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class, checkpoint = load_model(args.checkpoint, device)
    transform = make_predict_transform(checkpoint)
    paths = collect_images(args.input)
    if not paths:
        raise ValueError(f"No images found: {args.input}")

    rows = predict_paths(model, paths, transform, idx_to_class, device, args.topk)
    for row in rows[:20]:
        print(f"{row['path']} -> {row['pred_class']} ({row['confidence']:.4f})")
    if len(rows) > 20:
        print(f"... printed 20/{len(rows)} predictions")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
