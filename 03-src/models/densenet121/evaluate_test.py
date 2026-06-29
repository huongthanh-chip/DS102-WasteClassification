import json
import numpy as np
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "03-src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataloader import scan_image_folder, make_transforms, ImagePathDataset, DATASET_MEAN, DATASET_STD

# ─────────────────────────────────────────
# 0. CẤU HÌNH
# ─────────────────────────────────────────
SPLIT_DIR  = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
TEST_DIR = SPLIT_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "05-models" / "densenet121"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMAGE_SIZE = 224
FEATURE_DIM = 200

CLASSES = ["Electronics", "Glass", "Metal", "Organic", "Other", "Paper", "Plastic", "Textiles"]
NUM_CLASSES = len(CLASSES)

print(f"Device: {DEVICE}")
print(f"Test dir: {TEST_DIR}")


# ─────────────────────────────────────────
# 1. ĐỊNH NGHĨA MODEL (giống densenet121_svm.py)
# ─────────────────────────────────────────
class DenseNetSVMFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 200):
        super().__init__()
        backbone = densenet121(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier.in_features  # 1024
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.feat_200 = nn.Linear(512, feature_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop1(self.relu1(self.fc1(x)))
        feat = self.relu2(self.feat_200(x))
        if return_features:
            return feat
        return self.classifier(self.drop2(feat))


# ─────────────────────────────────────────
# 2. LOAD DATA TEST
# ─────────────────────────────────────────
def load_test_data():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")

    label_map = {cls: idx for idx, cls in enumerate(CLASSES)}
    paths, labels, _ = scan_image_folder(TEST_DIR, label_map=label_map)

    _, val_transform = make_transforms(
        image_size=IMAGE_SIZE,
        mean=DATASET_MEAN,
        std=DATASET_STD,
        augment_train=False,
    )

    ds = ImagePathDataset(paths, labels, transform=val_transform, return_path=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test set: {len(ds):,} images")
    return loader, np.array(labels)


# ─────────────────────────────────────────
# 3. DỰ ĐOÁN CNN
# ─────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader):
    """Evaluate DenseNet121 classifier head on the test split."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return labels, preds


# ─────────────────────────────────────────
# 4. VẼ CONFUSION MATRIX
# ─────────────────────────────────────────
def plot_confusion_matrix(labels, preds, title: str, save_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Saved -> {save_path}")


# ─────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("EVALUATE TEST SET")
    print("="*50)

    model_path = OUTPUT_DIR / "densenet121_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Run train_densenet121.py first to create the CNN checkpoint."
        )

    # 1. Load data
    print("\n--- Load test data ---")
    test_loader, true_labels = load_test_data()

    # 2. Load model
    print("\n--- Load model ---")
    model = DenseNetSVMFeatureExtractor(NUM_CLASSES, FEATURE_DIM)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"  Loaded: {model_path}")

    print("\n--- Run CNN inference ---")
    true_labels, cnn_preds = run_inference(model, test_loader)

    print("\n" + "="*50)
    print("DENSENET121 CNN RESULTS")
    print("="*50)
    cnn_acc = accuracy_score(true_labels, cnn_preds)
    report = classification_report(
        true_labels,
        cnn_preds,
        target_names=CLASSES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(true_labels, cnn_preds).tolist()

    print(f"  Accuracy: {cnn_acc:.4f}")
    print(classification_report(true_labels, cnn_preds, target_names=CLASSES, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(np.asarray(cm))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(model_path),
                "test_dir": str(TEST_DIR),
                "accuracy": float(cnn_acc),
                "classification_report": report,
                "confusion_matrix": cm,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Saved metrics -> {metrics_path}")

    print("\n--- Save confusion matrix ---")
    plot_confusion_matrix(
        true_labels, cnn_preds,
        title="Confusion Matrix - DenseNet121 CNN (Test)",
        save_path=OUTPUT_DIR / "confusion_matrix_test_cnn.png",
    )

    print("\nXong!")


if __name__ == "__main__":
    main()
