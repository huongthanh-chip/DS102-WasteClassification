# ConvNeXt-Tiny

Commands from the repository root:

```powershell
python 03-src\models\convnext_tiny\train_convnext_tiny.py --use-augmented-dir --epochs 50 --batch-size 32
python 03-src\models\convnext_tiny\evaluate_val.py
python 03-src\models\convnext_tiny\evaluate_test.py
python 03-src\models\convnext_tiny\extract_features.py --batch-size 64
```

Artifacts:

```text
05-models/convnext_tiny/
04-features/convnext_tiny/
```
