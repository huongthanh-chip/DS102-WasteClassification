#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SRC = PROJECT_ROOT / "03-src" / "data"
if str(DATA_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_SRC))

from dataloader import (
    DEFAULT_LABEL_MAP,
    build_dataloaders_from_train_val_dirs,
    load_label_map,
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "05-models" / "efficientnet_b0"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def create_efficientnet_b0(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.4,
) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
        except Exception:
            weights = None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_class_weights(info: dict, device: torch.device) -> torch.Tensor:
    weights = np.asarray(info["class_weights"], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss(
    loss_name: str,
    class_weights: torch.Tensor | None,
    gamma: float,
    label_smoothing: float,
):
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    raise ValueError("loss_name must be 'focal' or 'ce'.")


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool, desc: str):
    model.train(train)
    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []

    progress = tqdm(loader, desc=desc, leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        seen = len(all_targets)
        progress.set_postfix(
            loss=f"{total_loss / max(seen, 1):.4f}",
            acc=f"{accuracy_score(all_targets, all_preds):.4f}",
        )

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_targets, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }
    return metrics, all_targets, all_preds


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    best_val_f1: float,
    info: dict,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1_macro": best_val_f1,
        "label_map": info["label_map"],
        "idx_to_class": info["idx_to_class"],
        "image_size": info["image_size"],
        "mean": info["mean"],
        "std": info["std"],
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def write_history_csv(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_precision_macro",
        "train_recall_macro",
        "train_f1_macro",
        "train_precision_weighted",
        "train_recall_weighted",
        "train_f1_weighted",
        "val_loss",
        "val_acc",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_precision_weighted",
        "val_recall_weighted",
        "val_f1_weighted",
        "lr",
        "seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 for waste classification.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", choices=["focal", "ce"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Only used with --loss ce.",
    )
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = args.train_dir
    if train_dir is None:
        train_dir = args.split_dir / "train_augmented"
        if not train_dir.exists():
            train_dir = args.split_dir / "train"
    val_dir = args.val_dir or (args.split_dir / "val")

    train_loader, val_loader, info = build_dataloaders_from_train_val_dirs(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment_train=True,
        use_weighted_sampler=False,
        label_map_path=args.label_map,
    )

    model = create_efficientnet_b0(
        info["num_classes"],
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    class_weights = None if args.no_class_weights else get_class_weights(info, device)
    criterion = build_loss(
        args.loss,
        class_weights,
        args.focal_gamma,
        args.label_smoothing,
    )

    print(f"device     : {device}")
    print(f"train_dir  : {train_dir}")
    print(f"val_dir    : {val_dir}")
    print(f"classes    : {info['label_map']}")
    print(f"loss       : {args.loss}")
    print(f"optimizer  : {args.optimizer}")
    print(f"dropout    : {args.dropout}")
    print(f"weight_decay: {args.weight_decay}")
    if args.loss == "ce":
        print(f"label_smoothing: {args.label_smoothing}")
    print("scheduler  : ReduceLROnPlateau monitor=val_loss factor=0.5 patience=3")
    print(f"early_stop : monitor=val_macro_f1 patience={args.patience} min_delta={args.min_delta}")
    print("best point : val_macro_f1 -> best.pt")
    print("best loss  : val_loss -> best_loss.pt")
    print(f"batch_size : {args.batch_size}")
    print(f"epochs     : {args.epochs}")

    best_val_loss = float("inf")
    best_val_f1 = -float("inf")
    bad_epochs = 0
    history = []
    best_path = args.model_dir / "best.pt"
    best_loss_path = args.model_dir / "best_loss.pt"
    last_path = args.model_dir / "last.pt"
    history_csv_path = args.model_dir / "history.csv"
    history_json_path = args.model_dir / "history.json"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics, _, _ = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            desc=f"epoch {epoch:03d}/{args.epochs} train",
        )
        val_metrics, val_targets, val_preds = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            desc=f"epoch {epoch:03d}/{args.epochs} val",
        )
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        val_f1_macro = val_metrics["f1_macro"]
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_precision_weighted": train_metrics["precision_weighted"],
            "train_recall_weighted": train_metrics["recall_weighted"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_weighted": val_metrics["precision_weighted"],
            "val_recall_weighted": val_metrics["recall_weighted"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "lr": lr,
            "seconds": round(time.time() - start, 2),
        }
        history.append(row)
        write_history_csv(history_csv_path, history)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_f1={train_metrics['f1_macro']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"lr={lr:.2e}"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            best_val_f1,
            info,
            args,
        )
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            save_checkpoint(
                best_loss_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_val_f1,
                info,
                args,
            )
            print(f"  saved best loss -> {best_loss_path}")

        if val_f1_macro > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1_macro
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_val_f1,
                info,
                args,
            )
            print(f"  saved best f1 -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"early stopping at epoch {epoch} "
                    f"(val_macro_f1 did not improve for {args.patience} epochs)"
                )
                break

    args.model_dir.mkdir(parents=True, exist_ok=True)
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    idx_to_class = {int(k): v for k, v in info["idx_to_class"].items()}
    target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    print("\nValidation report from last epoch:")
    print(classification_report(val_targets, val_preds, target_names=target_names, digits=4))
    print("confusion_matrix:")
    print(confusion_matrix(val_targets, val_preds))


if __name__ == "__main__":
    main()
