"""
K-Means Clustering với CLIP Embeddings + UMAP Visualization
─────────────────────────────────────────────────────────────
Pipeline:
  1. Load ảnh từ cleaned_dataset
  2. Trích CLIP embeddings (ViT-B/32) — hiểu ngữ nghĩa ảnh
  3. L2-normalize
  4. K-Means k=8 trên không gian CLIP (512-D)
  5. UMAP 2D để visualize (giữ cấu trúc cục bộ tốt hơn PCA)
  6. Output: scatter UMAP, cluster grid, elbow+silhouette

Yêu cầu:
  pip install torch torchvision open-clip-torch umap-learn pillow \
              scikit-learn matplotlib numpy tqdm
"""

import os, sys, csv, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import open_clip

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import umap

warnings.filterwarnings("ignore")
9
# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DATASET_DIR       = "cleaned_dataset"
K                 = 8
BATCH_SIZE        = 64
OUTPUT_DIR        = "kmeans_output_clip"
RANDOM_STATE      = 42
MAX_GRID_IMGS     = 8
EXTENSIONS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# UMAP params
UMAP_N_NEIGHBORS  = 30     # tăng → cấu trúc toàn cục; giảm → cấu trúc cục bộ
UMAP_MIN_DIST     = 0.05   # giảm → cluster compact hơn
# ══════════════════════════════════════════════

COLORS = ["#e63946","#f4a261","#2a9d8f","#457b9d",
          "#e9c46a","#6d6875","#90be6d","#ff6b9d"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")


# ──────────────────────────────────────────────
# 1. LOAD PATHS
# ──────────────────────────────────────────────
def load_paths(folder):
    paths = []
    for ext in EXTENSIONS:
        paths.extend(Path(folder).rglob(f"*{ext}"))
        paths.extend(Path(folder).rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))
    if not paths:
        print(f"[ERROR] Không tìm thấy ảnh trong '{folder}'")
        sys.exit(1)
    print(f"[INFO] Tìm thấy {len(paths)} ảnh.")
    return paths


# ──────────────────────────────────────────────
# 2. LOAD CLIP MODEL
# ──────────────────────────────────────────────
def load_clip():
    print("[INFO] Đang load CLIP ViT-B/32 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(DEVICE)
    print("[INFO] CLIP loaded  (embedding dim: 512)")
    return model, preprocess


# ──────────────────────────────────────────────
# 3. TRÍCH CLIP EMBEDDINGS
# ──────────────────────────────────────────────
def extract_clip_embeddings(paths, model, preprocess):
    print("[INFO] Đang trích CLIP embeddings...")
    all_embs, valid_paths = [], []
    batch_tensors, batch_paths = [], []

    def process_batch():
        tensor = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda"):
            emb = model.encode_image(tensor)
            emb = emb.float().cpu().numpy()
        all_embs.append(emb)
        valid_paths.extend(batch_paths)

    for p in tqdm(paths, desc="  CLIP encode"):
        try:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(preprocess(img))
            batch_paths.append(str(p))
        except Exception as e:
            print(f"\n  [SKIP] {Path(p).name}: {e}")
            continue

        if len(batch_tensors) == BATCH_SIZE:
            process_batch()
            batch_tensors.clear()
            batch_paths.clear()

    if batch_tensors:
        process_batch()

    embeddings = np.vstack(all_embs)
    print(f"[INFO] Embeddings shape : {embeddings.shape}")
    return embeddings, valid_paths


# ──────────────────────────────────────────────
# 4. L2-NORMALIZE
# ──────────────────────────────────────────────
def preprocess_embeddings(embeddings):
    print("[INFO] L2-normalizing embeddings...")
    return normalize(embeddings, norm="l2")


# ──────────────────────────────────────────────
# 5. K-MEANS
# ──────────────────────────────────────────────
def run_kmeans(features):
    print(f"[INFO] Chạy K-Means k={K} ...")
    km = KMeans(n_clusters=K, init="k-means++", n_init=20,
                max_iter=500, random_state=RANDOM_STATE)
    labels = km.fit_predict(features)

    sil = silhouette_score(features, labels,
                           sample_size=min(5000, len(labels)),
                           random_state=RANDOM_STATE)
    print(f"  Inertia (WCSS)   : {km.inertia_:.2f}")
    print(f"  Silhouette Score : {sil:.4f}")
    for c in range(K):
        print(f"  Cluster {c}: {(labels==c).sum()} ảnh")
    return labels, km, sil


# ──────────────────────────────────────────────
# 6. UMAP 2D
# ──────────────────────────────────────────────
def run_umap(features):
    print(f"[INFO] Chạy UMAP 2D (n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}) — có thể mất vài phút...")

    # PCA 64D trước để tăng tốc UMAP trên dataset lớn
    n_pre = min(64, features.shape[1], features.shape[0] - 1)
    pca = PCA(n_components=n_pre, random_state=RANDOM_STATE)
    pre = pca.fit_transform(features)
    print(f"  → Pre-PCA: {features.shape[1]}D → {n_pre}D")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",          # tốt cho normalized embeddings
        random_state=RANDOM_STATE,
        verbose=True,
    )
    emb_2d = reducer.fit_transform(pre)
    print(f"[INFO] UMAP xong — shape: {emb_2d.shape}")
    return emb_2d


# ──────────────────────────────────────────────
# 7. VISUALIZE — UMAP Scatter
# ──────────────────────────────────────────────
def plot_umap_scatter(umap_2d, labels, sil):
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    for c in range(K):
        m = labels == c
        ax.scatter(umap_2d[m, 0], umap_2d[m, 1],
                   s=12, alpha=0.65, color=COLORS[c],
                   edgecolors="none", zorder=2,
                   label=f"Cluster {c}  (n={m.sum()})")

    ax.set_title(
        f"K-Means k=8 — CLIP ViT-B/32 Embeddings  (UMAP 2D)\n"
        f"Silhouette Score: {sil:.4f}",
        fontsize=15, color="white", fontweight="bold", pad=14)
    ax.set_xlabel("UMAP 1", color="#888", fontsize=11)
    ax.set_ylabel("UMAP 2", color="#888", fontsize=11)
    ax.tick_params(colors="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a3a")
    ax.grid(True, ls="--", alpha=0.08, color="white")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.25,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="#333",
              markerscale=2.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "umap_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  → {out}")


# ──────────────────────────────────────────────
# 8. VISUALIZE — Cluster Grid
# ──────────────────────────────────────────────
def load_thumb(path, size=128):
    try:
        return np.array(Image.open(path).convert("RGB").resize((size, size)))
    except:
        return np.zeros((size, size, 3), dtype=np.uint8)

def plot_cluster_grid(paths, labels):
    n_cols = MAX_GRID_IMGS
    rng = np.random.default_rng(RANDOM_STATE)

    fig, axes = plt.subplots(K, n_cols, figsize=(n_cols * 2.1, K * 2.3))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Ảnh mẫu theo Cluster — CLIP + K-Means (k=8)",
                 fontsize=16, color="white", fontweight="bold", y=1.01)

    for c in range(K):
        idxs = np.where(labels == c)[0]
        chosen = rng.choice(idxs, size=min(n_cols, len(idxs)), replace=False)

        for col in range(n_cols):
            ax = axes[c][col]
            ax.set_facecolor("#0d0d1a")
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(load_thumb(paths[chosen[col]]))
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(COLORS[c])
                    sp.set_linewidth(2.2)
            else:
                ax.set_visible(False)

        axes[c][0].set_ylabel(f"C{c}", color=COLORS[c], fontsize=13,
                              fontweight="bold", rotation=0,
                              labelpad=32, va="center")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cluster_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  → {out}")


# ──────────────────────────────────────────────
# 9. VISUALIZE — Elbow + Silhouette
# ──────────────────────────────────────────────
def plot_elbow_silhouette(features, k_range=range(2, 13)):
    print("[INFO] Vẽ Elbow + Silhouette chart...")
    wcss, sils = [], []
    for k in tqdm(k_range, desc="  Scanning k"):
        km = KMeans(n_clusters=k, init="k-means++", n_init=5,
                    max_iter=200, random_state=RANDOM_STATE)
        lbl = km.fit_predict(features)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(features, lbl,
                                     sample_size=min(3000, len(lbl)),
                                     random_state=RANDOM_STATE))

    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#777")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax.grid(True, ls="--", alpha=0.12, color="white")

    ax1.plot(ks, wcss, "o-", color="#e63946", lw=2.5,
             ms=8, mfc="white", mec="#e63946")
    ax1.axvline(K, color="#f4a261", ls="--", alpha=0.9, label=f"k={K} (hiện tại)")
    ax1.set_title("Elbow — WCSS", color="white", fontsize=14, fontweight="bold")
    ax1.set_xlabel("k", color="#aaa"); ax1.set_ylabel("WCSS", color="#aaa")
    ax1.legend(labelcolor="white", framealpha=0.2,
               facecolor="#1a1a2e", edgecolor="#333")

    best_k = ks[int(np.argmax(sils))]
    ax2.plot(ks, sils, "s-", color="#2a9d8f", lw=2.5,
             ms=8, mfc="white", mec="#2a9d8f")
    ax2.axvline(K, color="#f4a261", ls="--", alpha=0.9, label=f"k={K} (hiện tại)")
    ax2.axvline(best_k, color="#90be6d", ls=":", alpha=0.9,
                label=f"k tốt nhất = {best_k}")
    ax2.set_title("Silhouette Score", color="white", fontsize=14, fontweight="bold")
    ax2.set_xlabel("k", color="#aaa"); ax2.set_ylabel("Score", color="#aaa")
    ax2.legend(labelcolor="white", framealpha=0.2,
               facecolor="#1a1a2e", edgecolor="#333")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "elbow_silhouette.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    print(f"  → {out}")


# ──────────────────────────────────────────────
# 10. LƯU CSV + EMBEDDINGS
# ──────────────────────────────────────────────
def save_results(paths, labels, embeddings):
    # CSV nhãn
    out_csv = os.path.join(OUTPUT_DIR, "cluster_labels.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "cluster"])
        for p, l in zip(paths, labels):
            w.writerow([p, int(l)])
    print(f"  → {out_csv}")

    # Lưu embeddings để dùng lại (không cần encode lại lần sau)
    out_emb = os.path.join(OUTPUT_DIR, "clip_embeddings.npy")
    np.save(out_emb, embeddings)
    print(f"  → {out_emb}  (dùng lại lần sau, không cần encode lại)")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    paths              = load_paths(DATASET_DIR)
    model, preprocess  = load_clip()
    embeddings, valid_paths = extract_clip_embeddings(paths, model, preprocess)
    emb_norm           = preprocess_embeddings(embeddings)
    labels, km, sil    = run_kmeans(emb_norm)
    umap_2d            = run_umap(emb_norm)

    print("\n[INFO] Đang vẽ biểu đồ...")
    plot_umap_scatter(umap_2d, labels, sil)
    plot_cluster_grid(valid_paths, labels)
    plot_elbow_silhouette(emb_norm)
    save_results(valid_paths, labels, embeddings)

    print(f"\n✅ Xong! Kết quả trong '{OUTPUT_DIR}/'")
    print("   umap_scatter.png      — UMAP 2D scatter theo cluster")
    print("   cluster_grid.png      — lưới ảnh mẫu mỗi cluster")
    print("   elbow_silhouette.png  — kiểm tra k tối ưu")
    print("   cluster_labels.csv    — nhãn cluster từng ảnh")
    print("   clip_embeddings.npy   — embeddings để tái sử dụng")