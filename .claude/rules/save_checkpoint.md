# Artifact & Checkpoint Storage — MANDATORY

## Rule: ALL training outputs go to /mnt/artifacts-datai/

The artifacts disk is a **separate 1.5TB SSD** mounted at `/mnt/artifacts-datai/`. This disk persists independently from the GPU server — when the GPU instance is dropped, we keep the artifacts disk to backup and push to HuggingFace.

**NEVER save checkpoints, logs, or trained models to /mnt/forge-data/ or the project directory.**

## Directory Layout

```
/mnt/artifacts-datai/
├── checkpoints/{PROJECT_NAME}/    ← model checkpoints during training
├── models/{PROJECT_NAME}/         ← final trained/exported models
├── logs/{PROJECT_NAME}/           ← training logs, metrics CSV
├── exports/{PROJECT_NAME}/        ← ONNX, TorchScript, quantized exports
├── reports/{PROJECT_NAME}/        ← evaluation reports, benchmarks
└── tensorboard/{PROJECT_NAME}/    ← TensorBoard event files
```

A convenience symlink exists: `/mnt/forge-data/artifacts → /mnt/artifacts-datai/`

## How to Apply

### In Python training scripts:
```python
import os
PROJECT = "project_loki"
ARTIFACTS = "/mnt/artifacts-datai"
CHECKPOINT_DIR = f"{ARTIFACTS}/checkpoints/{PROJECT}"
LOG_DIR = f"{ARTIFACTS}/logs/{PROJECT}"
TB_DIR = f"{ARTIFACTS}/tensorboard/{PROJECT}"
EXPORT_DIR = f"{ARTIFACTS}/exports/{PROJECT}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
```

### In TOML configs:
```toml
[checkpoint]
output_dir = "/mnt/artifacts-datai/checkpoints/project_loki"

[logging]
log_dir = "/mnt/artifacts-datai/logs/project_loki"
tensorboard_dir = "/mnt/artifacts-datai/tensorboard/project_loki"
```

### In CLI commands:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m anima_MODULE.train \
  --output-dir /mnt/artifacts-datai/checkpoints/project_loki \
  --log-dir /mnt/artifacts-datai/logs/project_loki
```

## After Training — Push to HuggingFace

```bash
cd /mnt/artifacts-datai/models/project_loki
hf upload ilessio-aiflowlab/PROJECT_NAME . . --private
```

## Why Separate Disk?

1. **Persistence**: GPU server is ephemeral (3-day rental). Artifacts disk stays.
2. **Backup**: Easy to snapshot the entire 1.5TB disk and download.
3. **Cost**: We only pay for GPU compute time, not for storing results.
4. **Safety**: Training crash won't corrupt source code or models on forge-data.


## MANDATORY: Train / Val / Test Split

ALWAYS split data before training:
- Train: 80% — gradient updates
- Val: 10% — evaluate every N steps, track overfitting
- Test: 10% — held out, final evaluation ONLY

Log val_loss every epoch. If val_loss diverges from train_loss → early stop.
Save split indices for reproducibility.

