# DenseNet121

Commands from the repository root:

```powershell
python 03-src\models\densenet121\train_densenet121.py
python 03-src\models\densenet121\evaluate_test.py
```

Artifacts:

```text
05-models/densenet121/
```

The old DenseNet-specific raw feature extraction script is intentionally not
included in the main pipeline. Use the shared handcrafted feature pipeline:

```powershell
python 03-src\feature_engineering.py --use-augmented-train
```
