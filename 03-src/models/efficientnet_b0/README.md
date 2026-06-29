# EfficientNet-B0

EfficientNet-B0 scripts:

```powershell
python 03-src\models\efficientnet_b0\train_efficientnet.py --epochs 50 --batch-size 16 --pretrained
python 03-src\models\efficientnet_b0\evaluate_cnn.py --checkpoint 05-models\efficientnet_b0\best.pt
python 03-src\extract_cnn_features.py --use-augmented-train --checkpoint 05-models\efficientnet_b0\best.pt
```

Artifacts:

```text
05-models/efficientnet_b0/
04-features/cnn_features_efficientnet_b0.npz
```
