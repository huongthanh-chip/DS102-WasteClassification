# CNN model organization

This folder keeps the three CNN model families used in the project.

## Train all CNNs

Run the end-to-end CNN comparison pipeline from the repository root:

```powershell
python 03-src\train_all_cnns.py --epochs 50 --efficientnet-batch-size 16 --convnext-batch-size 32
```

The script trains and evaluates:

- EfficientNet-B0
- ConvNeXt-Tiny
- DenseNet121

## EfficientNet-B0

Primary scripts are still the project-level scripts because other modules import them:

- `03-src/train_cnn.py`
- `03-src/evaluate_cnn.py`
- `03-src/extract_cnn_features.py`

Model artifacts are saved under:

- `05-models/efficientnet_b0/`

## ConvNeXt-Tiny

ConvNeXt-Tiny is currently the best CNN among the recorded experiments.

- `convnext_tiny/train_convnext_tiny.py`
- `convnext_tiny/evaluate_val.py`
- `convnext_tiny/evaluate_test.py`
- `convnext_tiny/extract_features.py`

Model artifacts should be under:

- `05-models/convnext_tiny/`

CNN feature cache should be under:

- `04-features/convnext_tiny/`

## DenseNet121

DenseNet121 is kept as a CNN baseline. Its old raw feature extraction path
(`combined_features.py`: HOG/LBP/color inside the DenseNet folder) is not part
of the main pipeline anymore. The main handcrafted feature pipeline is:

- `03-src/feature_engineering.py`
- `04-features/handcrafted_features.npz`

DenseNet artifacts should be under:

- `05-models/densenet121/`

## Hybrid feature pipeline

Use the best CNN embedding plus handcrafted features for classical classifiers:

- `03-src/train_best_cnn_hybrid.py`

This script selects ConvNeXt-Tiny as the current best CNN based on recorded
test macro F1, extracts ConvNeXt embeddings, aligns them with handcrafted
features by image path, then trains:

- Softmax Regression
- Linear SVM
- Random Forest

Outputs:

- `04-features/best_cnn_hybrid/`
- `05-models/best_cnn_hybrid/`

The old root-level experiment folders `ConvNeXt_Tiny/` and `DenseNet121/` are
not part of the organized source tree anymore. Keep code under `03-src/models/`
and generated artifacts under `04-features/` or `05-models/`.
