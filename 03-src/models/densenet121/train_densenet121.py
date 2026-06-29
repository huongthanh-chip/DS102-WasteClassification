import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SRC = PROJECT_ROOT / "03-src" / "data"
if str(DATA_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_SRC))

from dataloader import build_dataloaders_from_train_val_dirs

# 0. CÀI ĐẶT CẤU HÌNH
SPLIT_DIR = PROJECT_ROOT / "01-data" / "Prepared_Merged_Clean_Split_60_20_20"
TRAIN_AUG_DIR = SPLIT_DIR / "train_augmented"           # tập train đã được tăng cường
VAL_DIR = SPLIT_DIR / "val"
OUTPUT_DIR = PROJECT_ROOT / "05-models" / "densenet121"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32             # số ảnh trong 1 batch (xử lý 32 ảnh/lần rồi mới cập nhật trọng số)
IMAGE_SIZE = 224            # Resize lại ảnh về 224x224 cho phù hợp mô hình DenseNet121 
# Số lần model xem lại dataset một lượt
EPOCHS_FT1 = 10             # Phase 1 
EPOCHS_FT2 = 20             # Phase 2 
FEATURE_DIM = 200           # Số chiều của vector đặc trưng cuối cùng (cho SVM và Kmeans)
RANDOM_SEED = 42
UNFREEZE_LAST_N = 50        # Ở Phase 2, chỉ mở khóa 50 layer cuối của DenseNet121
VAL_SPLIT = 0.25            # Tỉ lệ train/val/test thực tế đã được chia sẵn (thư mục Prepared_Merged_Clean_Split_60_20_20). Biến này không dùng trong script này.

# Optimizer (AdamW)
LR_P1, WD_P1 = 1e-3, 1e-4   # Phase 1: ít tham số (chỉ head) → decay nhẹ
LR_P2, WD_P2 = 1e-4, 5e-4   # Phase 2: nhiều tham số hơn (head + 50 layer backbone) → decay mạnh hơn, theo spec nhóm

# ReduceLROnPlateau (chung cho cả 2 phase)
LR_FACTOR = 0.5
LR_PATIENCE = 3

# EarlyStopping
ES_PATIENCE_P1 = 5          # Phase 1 
ES_PATIENCE_P2 = 8          # Phase 2
ES_MIN_DELTA = 1e-4

# Chọn thiết bị tính toán
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Cố định seed cho cả 2 thư viện
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────
# 1. XÂY DỰNG MODEL
# ─────────────────────────────────────────
class DenseNetSVMFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 200):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        backbone = densenet121(weights=weights)

        # DenseNet121 gốc có phần features (học hình ảnh) và phần classifier (phân loại 1000 lớp Imagenet)
        # Vì cần phân loại 8 lớp rác chứ không phải 1000 lớp nên bỏ Classifier đi
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier.in_features           # 1024 cho DenseNet121

        # DenseNet121 sau khi nhìn xong 1 tấm ảnh, nó tạo ra 1024 con số mô tả ảnh đó, quá nhiều và không cần thiết
        self.fc1 = nn.Linear(in_features, 512)                  # bỏ bớt thông tin ít quan trọng (1024 → 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        # Chắt lọc tiếp, giữ lại cái cốt lõi nhất
        self.feat_200 = nn.Linear(512, feature_dim)             # thu gọn tiếp → 200 (lấy features ở đây) (200 tham khảo từ bài báo)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        self.classifier = nn.Linear(feature_dim, num_classes)   # phân loại 8 lớp rác

    def forward(self, x, return_features: bool = False):
        x = self.features(x)                                    # ảnh (224×224×3) → DenseNet121 xử lý → feature map
        x = self.pool(x)                                        # feature map 3D → nén thành vector 1024 số
        x = torch.flatten(x, 1)                                 # đảm bảo shape phẳng [batch, 1024] (Đảm bảo dữ liệu có đúng dạng để tính toán)
        x = self.drop1(self.relu1(self.fc1(x)))                 # 1024→512, ReLU bỏ các số âm đi, Dropout ngẫu nhiên tắt 50% neuron để tránh model học vẹt
        feat = self.relu2(self.feat_200(x))                     # 512→200, đây là vector đặc trưng cần lấy

        if return_features:
            return feat                                         # Nếu đang ở chế độ lấy features thì trả về 200 số này luôn, không cần đi tiếp.

        out = self.classifier(self.drop2(feat))                 # Nếu đang train thì đi tiếp, ra 8 con số - mỗi số là xác suất ảnh thuộc 1 trong 8 lớp rác.
        return out      

    # Không cho phép cập nhật trọng số DenseNet121, giữ nguyên những gì nó đã học từ ImageNet
    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
    # DenseNet121 bên trong gồm nhiều block xếp chồng lên nhau
    def unfreeze_last_n_layers(self, n: int):
        children = list(self.features.children())
        for p in self.features.parameters():
            p.requires_grad = False                             # Đóng băng tất cả trước
        for layer in children[-n:]:                             # Lấy n block cuối cùng
            for p in layer.parameters():
                p.requires_grad = True                          # Chỉ mở khóa n block đó
# Chỉ cho phép n block cuối được điều chỉnh trong Phase 2. Các block đầu vẫn giữ nguyên vì chúng học những thứ cơ bản không cần thay đổi.


# ─────────────────────────────────────────
# 2. TRAIN 1 EPOCH / EVALUATE
# ─────────────────────────────────────────
# Chạy 1 lượt qua toàn bộ dataset
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    # → Xác định đang học (train) hay đang thi (val)
    model.train() if is_train else model.eval()

    # Khởi tạo bộ đếm
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_train):          # Chỉ tính gradient khi đang train
        for images, labels in loader:               # → Lấy từng batch 32 ảnh ra
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # → Cho model đoán, so với đáp án thật
            if is_train:
                optimizer.zero_grad()                  # Xóa kết quả tính toán của batch trước, nếu không xóa thì bị cộng dồn → sai

            outputs = model(images)                    # Cho 32 ảnh chạy qua model → ra dự đoán
            loss = criterion(outputs, labels)          # So dự đoán với đáp án thật → tính độ sai

            # → Nếu đang train → tự điều chỉnh lại
            # → Nếu đang val → bỏ qua dòng này, không điều chỉnh gì
            if is_train:
                loss.backward()                        # Tính "sai chỗ nào, sai bao nhiêu"
                optimizer.step()                       # Dựa vào đó điều chỉnh lại model

            preds = outputs.argmax(dim=1)              # lấy lớp có xác suất cao nhất làm dự đoán
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            # Ghi lại kết quả của batch này
            total_loss += loss.item() * images.size(0) # cộng dồn tổng loss ( tổng độ sai tích lũy)
            correct += (preds == labels).sum().item()  # đếm số dự đoán đúng
            total += images.size(0)                    # đếm tổng số ảnh đã xử lý

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    # → Trả về loss trung bình và accuracy của cả lượt
    return total_loss / total, correct / total, macro_f1


# ─────────────────────────────────────────
# 3. TRAIN MODEL (2 PHASE)
# ─────────────────────────────────────────
def train_model(model, train_loader, val_loader, class_weights, num_classes):
    # Thước đo độ sai - truyền class_weights vào để model chú ý hơn các lớp ít ảnh, tránh thiên vị lớp nhiều ảnh
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
        label_smoothing=0.1,
    )
    # Phase 1 - Backbone đóng băng: Giữ nguyên toàn bộ kinh nghiệm cũ của DenseNet121, chỉ dạy thêm phần phân loại rác
    print("PHASE 1: Train top layers (backbone frozen)")
    print(f"  lr={LR_P1}  weight_decay={WD_P1}  ES_patience={ES_PATIENCE_P1}")
    print("=" * 50)

    model.freeze_backbone()                                                     # Khóa DenseNet121 lại
    model.to(DEVICE)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P1, weight_decay=WD_P1,
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE)

    best_val_f1, bad_epochs = 0.0, 0

    for epoch in range(1, EPOCHS_FT1 + 1):
        # Train
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        # Val
        val_loss,   val_acc,   val_f1   = run_epoch(model, val_loader,   criterion)


        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)                                                # Kiểm tra có cần giảm tốc độ học không
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  [LR] Giảm learning rate: {prev_lr:.2e} -> {new_lr:.2e}")

        print(f"[P1 {epoch:02d}/{EPOCHS_FT1}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + ES_MIN_DELTA:
            best_val_f1 = val_f1
            bad_epochs = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_phase1.pt")       # Lưu model tốt nhất
        else:
            bad_epochs += 1
            if bad_epochs >= ES_PATIENCE_P1:
                print(f"  EarlyStopping Phase 1 tại epoch {epoch}")             # Dừng sớm nếu không tiến bộ
                break

    # Load lại model tốt nhất
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_phase1.pt", map_location=DEVICE))
    print(f"  Best val_macro_f1 Phase 1: {best_val_f1:.4f}")

    # Phase 2 - fine-tune 50 layer cuối: Tinh chỉnh lại 50 layer cuối để phù hợp hơn với bài toán phân loại rác 
    print(f"PHASE 2: Fine-tune {UNFREEZE_LAST_N} layers cuối")
    print(f"  lr={LR_P2}  weight_decay={WD_P2}  ES_patience={ES_PATIENCE_P2}")
    print("=" * 50)

    model.unfreeze_last_n_layers(UNFREEZE_LAST_N)       # mở khóa 50 layer cuối của DenseNet121
    trainable = sum(p.requires_grad for p in model.features.parameters())
    total_p = sum(1 for _ in model.features.parameters())
    print(f"  Layers unfreeze: {trainable}/{total_p}")

    # Tốc độ học nhỏ hơn Phase 1 - chỉ tinh chỉnh nhẹ, không phá vỡ những gì đã học
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P2, weight_decay=WD_P2,
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE)

    best_val_f1, bad_epochs = 0.0, 0

    for epoch in range(1, EPOCHS_FT2 + 1):
        # Train
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        # Val
        val_loss,   val_acc,   val_f1   = run_epoch(model, val_loader,   criterion)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  [LR] Giảm learning rate: {prev_lr:.2e} -> {new_lr:.2e}")

        print(f"[P2 {epoch:02d}/{EPOCHS_FT2}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + ES_MIN_DELTA:
            best_val_f1 = val_f1
            bad_epochs  = 0
            # lưu model tốt nhất
            torch.save(model.state_dict(), OUTPUT_DIR / "best_final.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= ES_PATIENCE_P2:
                print(f"  EarlyStopping Phase 2 tại epoch {epoch}")
                break
    
    # Load lại model tốt nhất của Phase 2
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_final.pt", map_location=DEVICE))
    print(f"  Best val_macro_f1 Phase 2: {best_val_f1:.4f}")
    return model

# ─────────────────────────────────────────
# 4. TRÍCH XUẤT ĐẶC TRƯNG CNN
# ─────────────────────────────────────────
@torch.no_grad()
def extract_features_with_paths(model, dataset_dir: Path, classes: list, batch_size: int = 32, image_size: int = 224):
    """
    Trích xuất CNN feature theo ĐÚNG MỘT danh sách đường dẫn ảnh duy nhất,
    rồi trả về luôn list path đó — để combined_features.py dùng lại CHÍNH XÁC
    cùng list path này khi tính HOG/LBP/Color, tránh lệch thứ tự ảnh.
    """
    from torch.utils.data import DataLoader
    from dataloader import scan_image_folder, make_transforms, ImagePathDataset, DATASET_MEAN, DATASET_STD

    # 1. Thu thập danh sách ảnh CỐ ĐỊNH — dùng lại scan_image_folder() của dataloader.py
    #    để đồng bộ cách quét ảnh (đệ quy, đủ extension, sorted) với phần train.
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    paths, labels, _ = scan_image_folder(dataset_dir, label_map=label_map)

    # 2. SỬA: dùng lại make_transforms() với ĐÚNG DATASET_MEAN/STD đã dùng lúc train
    #    (trước đây tự viết transform riêng với mean/std chuẩn ImageNet, khác với
    #    mean/std riêng của dataset dùng khi train model → gây lệch chuẩn hóa giữa
    #    lúc train và lúc trích feature, ảnh hưởng đến chất lượng SVM/K-Means)
    _, val_transform = make_transforms(
        image_size=image_size, mean=DATASET_MEAN, std=DATASET_STD, augment_train=False
    )

    ds = ImagePathDataset(paths, labels, transform=val_transform, return_path=False)
    # shuffle=False BẮT BUỘC - đảm bảo thứ tự feature khớp đúng thứ tự `paths`
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_feats, all_labels = [], []
    for images, lbls in loader:
        feats = model(images.to(DEVICE), return_features=True)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(np.array(lbls))

    feats_arr = np.concatenate(all_feats)
    labels_arr = np.concatenate(all_labels)
    return feats_arr, labels_arr, [str(p) for p in paths]   # ← trả thêm paths để dùng tiếp ở bước raw feature


# ─────────────────────────────────────────
# 5. TRAIN SVM
# Train SVM để phân loại rác dựa trên 200 features vừa trích xuất.
# ─────────────────────────────────────────
def train_svm(train_features, train_labels, val_features, val_labels, classes):
    print("TRAIN SVM")
    print("=" * 50)

    # Chuẩn hóa features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_val = scaler.transform(val_features)

    # GridSearch - tìm cấu hình SVM tốt nhất
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([train_labels, val_labels])
    split_idx = np.concatenate([
        np.full(len(X_train), -1),      # -1 = fold train
        np.full(len(X_val),    0),      #  0 = fold đánh giá
    ])

    print("  Đang Grid Search (12 tổ hợp)...")
    gs = GridSearchCV(
        SVC(class_weight="balanced", random_state=RANDOM_SEED),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        cv=PredefinedSplit(split_idx),
        scoring="accuracy", n_jobs=-1, refit=False, verbose=1,
    )
    gs.fit(X_all, y_all)
    print(f"  Best params: {gs.best_params_}")

    # Train SVM - Sau khi biết cấu hình tốt nhất, train SVM chính thức
    svm = SVC(**gs.best_params_, class_weight="balanced",
              random_state=RANDOM_SEED, probability=True)
    svm.fit(X_train, train_labels)

    # Đánh giá kết quả
    val_pred = svm.predict(X_val)
    print(f"\n  Val Accuracy (SVM): {accuracy_score(val_labels, val_pred):.4f}")
    print(classification_report(val_labels, val_pred, target_names=classes))

    return svm, scaler


# ─────────────────────────────────────────
# 6. LƯU KẾT QUẢ
# ─────────────────────────────────────────
def save_outputs(model, svm, scaler, train_feats, train_labels, val_feats, val_labels,
                 train_paths, val_paths):
    torch.save(model.state_dict(), OUTPUT_DIR / "densenet121_model.pt")
    joblib.dump(svm, OUTPUT_DIR / "svm_classifier.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "svm_scaler.pkl")

    np.save(OUTPUT_DIR / "train_features.npy", train_feats)
    np.save(OUTPUT_DIR / "train_labels.npy", train_labels)
    np.save(OUTPUT_DIR / "val_features.npy", val_feats)
    np.save(OUTPUT_DIR / "val_labels.npy", val_labels)

    # Lưu đường dẫn ảnh THEO ĐÚNG THỨ TỰ feature — combined_features.py sẽ dùng lại
    # để tính HOG/LBP/Color khớp 1-1 với từng dòng feature, tránh lệch thứ tự ảnh.
    with open(OUTPUT_DIR / "train_paths.json", "w") as f:
        json.dump(train_paths, f)
    with open(OUTPUT_DIR / "val_paths.json", "w") as f:
        json.dump(val_paths, f)

    all_feats = np.vstack([train_feats, val_feats])
    all_labels = np.concatenate([train_labels, val_labels])
    np.save(OUTPUT_DIR / "all_features.npy", all_feats)
    np.save(OUTPUT_DIR / "all_labels.npy", all_labels)

    print(f"\n  train_features : {train_feats.shape}  (paths: {len(train_paths)})")
    print(f"  val_features   : {val_feats.shape}  (paths: {len(val_paths)})")
    print(f"  all_features   : {all_feats.shape}  <- cho K-Means")
    print(f"  Tất cả lưu vào: {OUTPUT_DIR}/")


# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────
def main():
    print("DenseNet121 + SVM (PyTorch) — v3 (đồng nhất config nhóm)")
    print(f"  Train: {TRAIN_AUG_DIR}")
    print(f"  Val  : {VAL_DIR}")
    print(f"  Output: {OUTPUT_DIR}\n")

    if not TRAIN_AUG_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Chưa thấy {TRAIN_AUG_DIR} hoặc {VAL_DIR}.\n"
            f"Kiểm tra lại thư mục {SPLIT_DIR} đã có đủ 4 folder "
            "(train, train_augmented, val, test) và split_manifest.csv chưa."
        )

    # 1. Load data: train_augmented (đã augment) + val (sạch, không augment)
    train_loader, val_loader, info = build_dataloaders_from_train_val_dirs(
        train_dir=TRAIN_AUG_DIR, val_dir=VAL_DIR,
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        augment_train=False, use_weighted_sampler=False,
    )
    classes = [info["idx_to_class"][i] for i in sorted(info["idx_to_class"])]
    num_classes = info["num_classes"]
    class_weights = info["class_weights"]

    print(f"  Classes      : {classes}")
    print(f"  Train size   : {info['train_size']}")
    print(f"  Val size     : {info['val_size']}")
    print(f"  Class weights: {np.round(class_weights, 3).tolist()}")

    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(info["label_map"], f, indent=2)

    # 2. Build model
    model = DenseNetSVMFeatureExtractor(num_classes, FEATURE_DIM)

    # 3. Train (Phase 1 + Phase 2)
    model = train_model(model, train_loader, val_loader, class_weights, num_classes)

    # 4. Trích xuất CNN features
    print("TRÍCH XUẤT CNN FEATURES (kèm đường dẫn ảnh, thứ tự cố định)")
    print("=" * 50)
    train_feats, train_labels, train_paths = extract_features_with_paths(
        model, TRAIN_AUG_DIR, classes, BATCH_SIZE, IMAGE_SIZE)
    val_feats, val_labels, val_paths = extract_features_with_paths(
        model, VAL_DIR, classes, BATCH_SIZE, IMAGE_SIZE)

    # 5. Train SVM
    svm, scaler = train_svm(train_feats, train_labels, val_feats, val_labels, classes)

    # 6. Lưu tất cả
    print("\n" + "=" * 50)
    print("LƯU KẾT QUẢ")
    print("=" * 50)
    save_outputs(model, svm, scaler, train_feats, train_labels, val_feats, val_labels, train_paths, val_paths)
    print("\nXong!")


if __name__ == "__main__":
    main()
