# PRD-04: Training Pipeline (Optimization Pipeline)

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-01, PRD-02, PRD-03

## Objective

Build the full optimization pipeline: synthetic data generation, SLOT optimization loop,
checkpointing, logging. Note: TAG "training" is per-scene optimization, not gradient-based
NN training. However, we also support an optional learned initialization network.

## Deliverables

### 1. Synthetic Data Generator (`src/tag/dataset.py`)

```python
class SyntheticThermalDataset(Dataset):
    """
    Generates hyperspectral thermal scenes from:
      - Known temperature maps (random, structured, from IR images)
      - Material emissivity spectra (from spectral libraries or parametric)
      - View factors (random, smooth)
      - Sensor noise (NEDT ~ 30 mK)

    Returns: (S_obs, S_sky, S_ground, T_gt, e_gt, V_gt)
    """
```

### 2. Real IR Data Adapter (`src/tag/dataset.py`)

```python
class IRImageDataset(Dataset):
    """
    Wraps single-band IR images (e.g., NUAA-SIRST) for testing the pipeline.
    Simulates multi-spectral from single-band using parametric emissivity model.
    """
```

### 3. Optimization Runner (`src/tag/train.py`)

```python
class SLOTRunner:
    """
    Runs SLOT optimization on a scene:
      1. Initialize T, beta, V
      2. Run L-BFGS or projected GD for N iterations
      3. Log convergence: residual, smoothness, constraint violations
      4. Save results: T_map, e_cube, V_map, texture_image
    """
```

### 4. Checkpointing
- Save intermediate optimization state every N iterations
- Save best result (lowest objective)
- Resume from checkpoint
- Output to `/mnt/artifacts-datai/checkpoints/project_tag/`

### 5. Logging
- Per-iteration: objective, residual, smoothness, constraint violation
- Per-scene: convergence plot, runtime
- TensorBoard: objective curves, temperature/emissivity visualizations

### 6. Config-Driven
- All parameters in TOML (lambda, K, max_iter, tolerance, noise level)
- `configs/paper.toml` matches paper settings exactly

## Acceptance Criteria
- [ ] Synthetic dataset generates valid hyperspectral thermal cubes
- [ ] SLOT recovers T within 1K, emissivity within 0.05 on noise-free synthetic data
- [ ] Convergence within 100 iterations on 64x64 synthetic scene
- [ ] Checkpoint save/load cycle works
- [ ] Logs written to /mnt/artifacts-datai/logs/project_tag/
