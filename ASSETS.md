# ASSETS.md — TAG Module Asset Inventory

> Last updated: 2026-04-05

## Datasets Needed

### Primary (Required for core pipeline)

| Asset | Type | Size Est. | Path | Status |
|-------|------|-----------|------|--------|
| Synthetic LWIR hyperspectral | Generated | ~2 GB | /mnt/forge-data/datasets/tag_synthetic_lwir/ | NOT CREATED |
| DARPA Invisible Headlights | Thermal scenes | ~5 GB | /mnt/forge-data/datasets/darpa_headlights/ | NOT AVAILABLE (request-only) |
| NUAA-SIRST (IR small targets) | Infrared images | On disk | /mnt/forge-data/datasets/nuaa_sirst_yolo/ | AVAILABLE |

### Synthetic Data Generation

Since TAG requires **hyperspectral thermal data** (67 spectral bands, 870-1269 cm^-1)
that is not publicly available, we generate synthetic data:

1. **Planck-based synthesis**: Generate multi-spectral thermal cubes from temperature maps
   + material emissivity libraries (ECOSTRESS, ASTER spectral libraries)
2. **LWIR simulation**: Use known emissivity spectra of common materials (skin, fabric,
   metal, vegetation) to create realistic hyperspectral thermal scenes
3. **Noise model**: Add sensor noise (NEDT ~ 30 mK) and atmospheric effects

### Reference Spectral Libraries (for synthesis)

| Library | Source | Path |
|---------|--------|------|
| ECOSTRESS spectral library | NASA/JPL | /mnt/forge-data/datasets/tag_spectral_libs/ecostress/ |
| ASTER spectral library | USGS | /mnt/forge-data/datasets/tag_spectral_libs/aster/ |

**Status**: NOT DOWNLOADED — needed for synthetic data generation.

## Pretrained Models

| Model | Use | Path | Status |
|-------|-----|------|--------|
| MediaPipe Face Mesh | Downstream eval (Task II) | pip install mediapipe | NOT INSTALLED |
| MTCNN | Downstream eval (Task III) | pip install facenet-pytorch | NOT INSTALLED |
| FER | Downstream eval (Task III) | pip install fer | NOT INSTALLED |

**Note**: TAG itself has **no pretrained weights** — SLOT is a physics-based optimization,
not a learned model. The "models" above are for downstream task evaluation only.

## Shared Infrastructure

### CUDA Kernels (from /mnt/forge-data/shared_infra/)

| Kernel | Relevance | Path |
|--------|-----------|------|
| Fused image preprocess (#38) | Image normalization | cuda_extensions/fused_image_preprocess/ |

TAG is primarily CPU-bound (B-spline optimization per pixel). GPU acceleration is used
for batched Planck function evaluation and matrix operations.

### Existing Datasets on Disk

| Dataset | Relevance | Path |
|---------|-----------|------|
| NUAA-SIRST | IR imagery for testing pipeline | /mnt/forge-data/datasets/nuaa_sirst_yolo/ |

## Output Paths

| Type | Path |
|------|------|
| Checkpoints | /mnt/artifacts-datai/checkpoints/project_tag/ |
| Logs | /mnt/artifacts-datai/logs/project_tag/ |
| Exports | /mnt/artifacts-datai/exports/project_tag/ |
| Reports | /mnt/artifacts-datai/reports/project_tag/ |
| TensorBoard | /mnt/artifacts-datai/tensorboard/project_tag/ |

## Downloads Needed

```bash
# Spectral libraries (if reproducing synthetic data)
# ECOSTRESS: https://speclib.jpl.nasa.gov/
# ASTER: https://speclib.jpl.nasa.gov/

# No model downloads needed — TAG is optimization-based
# Downstream eval models install via pip (mediapipe, facenet-pytorch, fer)
```

## Hardware Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU VRAM | 2-8 GB | Batched Planck + matrix ops; not NN training |
| CPU | High single-thread perf | B-spline optimization is CPU-heavy |
| RAM | 8+ GB | Hyperspectral cubes: 320x256x67 x float32 ~ 22 MB per frame |
| Disk | ~10 GB | Synthetic data + outputs |
