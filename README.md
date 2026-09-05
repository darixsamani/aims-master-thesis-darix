# Computer Vision for Multi-Category African Plum (Safou) Defect Detection

**Master's Thesis — African Institute for Mathematical Sciences (AIMS), Rwanda**

| | |
|---|---|
| **Author** | Darix Samani Siewe |
| **Program** | MSc in Mathematical Sciences with major in Data Science |
| **Institution** | African Institute for Mathematical Sciences (AIMS), Rwanda |

---

## Summary

This thesis benchmarks **23 deep learning architecture configurations** — spanning classic CNNs, lightweight mobile networks, modern convolutional designs, and a Vision Transformer — on the task of **multi-category defect classification** for the African Plum, locally known as *Safou* (*Dacryodes edulis*). The goal is an automated alternative to manual post-harvest quality inspection, which is slow, inconsistent, and hard to scale across Cameroon's plum-growing regions.

The full pipeline — exploratory data analysis, preprocessing, transfer learning, training, and evaluation — is implemented end-to-end in a single Jupyter notebook: [`code_thesis_darix_aims.ipynb`](./code_thesis_darix_aims.ipynb).

> **Workflow at a glance:** load and explore the dataset (class balance, sample-image grids) → build train/val transform pipelines → fine-tune each architecture via transfer learning → evaluate on Accuracy, Precision, Recall, F1 (weighted), and F1 (macro) → export training curves and confusion matrices for every completed run.

---

## Dataset

The **African Plum (Safou) Defect Detection Dataset**, collected across plum-growing regions of Cameroon and [published in *Data in Brief* (2025)](https://dx.doi.org/10.1016/j.dib.2025.111351), contains **4,507 labeled images** across six defect categories.

| Category | Images | Label |
|---|---:|---|
| `bruised` | 319 | defective |
| `cracked` | 162 | defective |
| `rotten` | 720 | defective |
| `spotted` | 759 | defective |
| `unaffected` | 1,721 | unaffected |
| `unripe` | 826 | unripe |
| **Total** | **4,507** | |

- **Split:** 3,831 training images / 676 validation images (85 / 15, seeded for reproducibility)
- **Preprocessing:** resize to 224 × 224, normalize with ImageNet mean/std
- **Augmentation (train only):** random horizontal flip, random vertical flip, color jitter (brightness/contrast/saturation = 0.2, hue = 0.05)

> ⚠️ Augmentation is applied to the training split only — the validation pipeline uses resize + normalize exclusively, to keep evaluation metrics uncorrupted.

---

## Repository Structure

```
aims-master-thesis-darix/
├── code_thesis_darix_aims.ipynb     # End-to-end pipeline: EDA → preprocessing → training → evaluation
├── data/
│   ├── README.md                    # Dataset documentation and citation
│   ├── organized_plums_data_new.csv # image_id, label, defect_type
│   └── african_plums/               # Source images, one subfolder per category
│       ├── bruised/  cracked/  rotten/
│       └── spotted/  unaffected/  unripe/
├── outputs/                         # Training curves + confusion matrix per completed model
├── african_plums_preview.png        # Sample-image grid, one row per category
├── distribution.png / distribution.pdf  # Class distribution plots
├── best_<model>.pth                 # Saved checkpoints (best validation accuracy per run)
└── .gitattributes
```

---

## Methodology

### Environment

The notebook is built on **PyTorch** / **torchvision**, with `pandas`, `matplotlib`, `scikit-learn`, and `Pillow` for data handling, evaluation, and visualization. Environment setup uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency installation — see Section 1 of the notebook for the full setup guide.

### Architectures Evaluated

Each architecture is adapted for 6-class classification by replacing its final classification layer, then fine-tuned end-to-end via transfer learning. All models share the same training recipe:

| Setting | Value |
|---|---|
| Loss | Cross-Entropy |
| Optimizer | AdamW (lr = 3e-4, weight decay = 1e-4) |
| Scheduler | Cosine annealing (T_max = 20) |
| Epochs | Up to 30, early stopping (patience = 5) |
| Checkpointing | Best validation accuracy |

<details>
<summary><strong>23 configurations across two model families</strong> (click to expand)</summary>

**EfficientNet family (11):** B0, B1, B2, B3, B4, B5, B6, B7, V2-S, V2-M, V2-L

**Classic, mobile & transformer models (12):** AlexNet, VGG16, VGG19, ResNet18, ResNet50, DenseNet121, MobileNetV2, MobileNetV3-Large, ShuffleNetV2, SqueezeNet1.1, ConvNeXt-Tiny, ViT-B/16

</details>

> ℹ️ This run was executed on **CPU**, with per-epoch times ranging from ~2 minutes (ShuffleNetV2, SqueezeNet) to over an hour (ViT-B/16). Under this constraint, the three largest EfficientNet variants (B5–B7) and the EfficientNet-V2 family (S/M/L) did not complete training within the thesis timeline; the remaining **17 configurations** finished and produced a saved checkpoint, of which **15** also have full training curves and a confusion matrix exported to [`outputs/`](./outputs).

### Evaluation Metrics

Every completed run is scored on: **Accuracy**, **Precision** (weighted), **Recall** (weighted), **F1 (weighted)**, and **F1 (macro)** — the macro F1 in particular surfaces performance on the minority classes (`cracked`, `bruised`) that a weighted average can mask.

---

## Sample Results

Final validation metrics for the runs whose summary was captured in the saved notebook:

| Model | Accuracy | Precision (w) | Recall (w) | F1 (weighted) | F1 (macro) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| **EfficientNet-B0** | **0.7071** | 0.6413 | 0.7071 | **0.6674** | **0.5501** | 16 |
| EfficientNet-B3 | 0.6953 | 0.6329 | 0.6953 | 0.6564 | 0.5386 | 19 |
| EfficientNet-B1 | 0.6849 | 0.6120 | 0.6849 | 0.6405 | 0.4999 | 19 |
| EfficientNet-B2 | 0.6746 | 0.6036 | 0.6746 | 0.6238 | 0.4830 | 19 |
| AlexNet | 0.6583 | 0.5917 | 0.6583 | 0.6195 | 0.4967 | 17 |

> Full per-model training curves and confusion matrices — including for ResNet18/50, DenseNet121, MobileNetV2/V3, ShuffleNetV2, SqueezeNet1.1, ConvNeXt-Tiny, VGG19, and ViT-B/16 — are available in [`outputs/`](./outputs).

---

## Getting Started

1. **Install dependencies** (see notebook Section 1 for the full `uv`-based setup):
   ```bash
   pip install uv
   uv tool install jupyterlab --with torch --with torchvision --with pandas \
     --with matplotlib --with scipy --with pillow --with scikit-learn
   ```
2. **Launch JupyterLab:**
   ```bash
   jupyter-lab
   ```
3. **Open** `code_thesis_darix_aims.ipynb` and run cells sequentially. Dataset paths are centralized in one configuration block (Section 2.2) — update `ROOT` and `CSV_FILE` if your local folder layout differs.
4. **Load a trained model** directly from a checkpoint, e.g.:
   ```python
   import torch
   from torchvision.models import efficientnet_b0
   import torch.nn as nn

   model = efficientnet_b0(weights=None)
   model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
   model.load_state_dict(torch.load("best_efficientnet_bo.pth", map_location="cpu"))
   model.eval()
   ```

---

## Limitations & Future Work

- **Compute:** all reported runs were trained on CPU; a GPU would likely close the gap for the still-untrained EfficientNet-B5–B7 and EfficientNet-V2 configurations and allow longer schedules for the rest.
- **Class imbalance:** `unaffected` (1,721 images) outweighs `cracked` (162 images) by more than 10×; macro-F1 is reported alongside weighted metrics for this reason, and class-balanced sampling or loss re-weighting is a natural next step.
- **Scope:** the notebook currently benchmarks single-label multi-class classification; extending to defect localization (bounding boxes / segmentation) would better support automated sorting lines.

---

## Citation

If you use this work, please cite it as:

```bibtex
@mastersthesis{samanisiewe2026safou,
  author  = {Samani Siewe, Darix},
  title   = {Computer Vision for Multi-Category African Plum (Safou) Defect Detection},
  school  = {African Institute for Mathematical Sciences (AIMS) Rwanda},
  year    = {2026},
  month   = jun,
  address = {Kigali, Rwanda},
  doi     = {10.13140/RG.2.2.24966.54082}
}
```

The underlying image dataset should be cited separately, per its own [documentation](./data/README.md):

```
Arnaud Nguembang Fadja. (2024). African Plums Dataset. Version 1.0. Kaggle.
```