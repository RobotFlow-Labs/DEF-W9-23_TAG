# Rule: ALWAYS Check Shared CUDA Infrastructure Before Building

## RULE
Before building ANY CUDA kernel or compiling ANY extension, CHECK if it already exists in shared_infra.
NEVER rebuild what is already compiled.

## SHARED CUDA KERNELS — /mnt/forge-data/shared_infra/cuda_extensions/
16 pre-compiled kernels (py3.11, cu128, sm_89):
- Gaussian rasterizer, Deformable attention, EAA renderer
- Trilinear voxelizer (304x), Batch voxelizer (11x)
- SE(3) transform (30x), Batched 3D IoU (15x)
- Depth projection (5.4x), Grid warp+sample (43.5x)
- Sparse upsample (6.4x), FPS (7.2x)
- Sparse Conv3D, Vectorized NMS, Scatter aggregate
- Vector quantization

Install: uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl

## PRE-COMPUTED DATASET CACHES — /mnt/forge-data/shared_infra/datasets/
565GB of pre-extracted features. DO NOT re-compute:
- nuScenes voxels (163GB), DINOv2 features (140GB)
- KITTI voxels (117GB), SERAPHIM VLM (53GB)
- COCO HDINO (25GB), COCO DINOv2 (9.9GB), COCO SAM2 (9.8GB)

## READ THE MAP
Full paths: /mnt/forge-data/shared_infra/MAP.md (also at .claude/MAP.md in your module)

## DO NOT
- DO NOT compile gaussian rasterizer — it exists
- DO NOT compile deformable attention — it exists
- DO NOT re-extract DINOv2 features — cached
- DO NOT re-voxelize nuScenes/KITTI — cached
- DO NOT build simple-knn — wheel exists

## HOW TO USE CACHED DATA
When training, check if your dataset has pre-computed features BEFORE extracting them yourself:

```python
import os
CACHE_DIR = "/mnt/forge-data/shared_infra/datasets/"

# Check available caches
caches = os.listdir(CACHE_DIR)
print("Available caches:", caches)

# Example: load COCO DINOv2 features instead of running DINOv2
if "coco_dinov2_features" in caches:
    features = torch.load(f"{CACHE_DIR}/coco_dinov2_features/features.pt")
    # Use features directly — skip DINOv2 forward pass
```

Available caches (565GB):
- coco_dinov2_features (9.9GB) — use instead of running DINOv2 on COCO
- coco_hdino_cache (25GB) — use instead of running H-DINO on COCO  
- coco_sam2_embeddings (9.8GB) — use instead of running SAM2 on COCO
- nuscenes_voxels (163GB) — use instead of voxelizing nuScenes
- nuscenes_dinov2_features (140GB) — use instead of running DINOv2 on nuScenes
- kitti_voxel_cache (117GB) — use instead of voxelizing KITTI
- seraphim_vlm_features (53GB) — use instead of running VLM on SERAPHIM
- kitti_dinov2_features (2.8GB) — use instead of running DINOv2 on KITTI

ALWAYS check this list before computing features. Loading a cache = instant. Computing = hours.

## UNPACKING DATA — ALWAYS USE SHARED LOCATION
When you unzip, untar, or extract ANY dataset, ALWAYS extract to:
/mnt/forge-data/shared_infra/datasets/{dataset_name}/

NEVER unpack data inside your module folder or anywhere else on forge-data.
forge-data disk is FULL (2.5TB). Shared datasets go to shared location.

If you need temporary space for extraction, use /mnt/train-data/tmp/ then move to shared_infra.
