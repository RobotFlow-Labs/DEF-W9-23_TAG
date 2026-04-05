# PRD-05: Evaluation

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-02, PRD-04

## Objective

Implement evaluation metrics matching Table 1 of the paper and downstream task evaluation.

## Deliverables

### 1. Image Quality Metrics (`src/tag/evaluate.py`)

Reproduce Table 1 metrics:
- **EN** (Information Entropy): -sum(p * log2(p)) over pixel histogram
- **AG** (Average Gradient): mean of sqrt(dx^2 + dy^2) / 2
- **SF** (Spatial Frequency): sqrt(RF^2 + CF^2) where RF/CF are row/col frequencies
- **SD** (Standard Deviation): std of pixel values

### 2. Reconstruction Metrics

- **RMSE_T**: Root mean squared error of temperature recovery
- **MAE_e**: Mean absolute error of emissivity recovery
- **RMSE_V**: Root mean squared error of view factor recovery
- **Spectral Angle Mapper (SAM)**: Angle between true and recovered emissivity spectra

### 3. Downstream Task Evaluation

- **Task I**: Colorization quality (SSIM, PSNR of colorized output vs reference)
- **Task II**: 3D mesh alignment (MediaPipe face mesh success rate, landmark RMSE)
- **Task III**: Emotion recognition (MTCNN detection rate, FER accuracy)

### 4. Evaluation Script (`scripts/evaluate.py`)

```bash
uv run python scripts/evaluate.py \
  --input results/decomposition/ \
  --ground-truth data/synthetic/gt/ \
  --output /mnt/artifacts-datai/reports/project_tag/
```

### 5. Comparison with HADAR Baseline

Run both SLOT and HADAR on same scenes, report all metrics side-by-side.

## Acceptance Criteria
- [ ] EN/AG/SF/SD metrics computed correctly on known test images
- [ ] SLOT outperforms raw IR on all 4 metrics (matches paper trend)
- [ ] Reconstruction RMSE_T < 2K on synthetic data with 30 mK noise
- [ ] Evaluation report generated as markdown + JSON
