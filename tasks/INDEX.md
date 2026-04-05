# Tasks Index — TAG Module

> Last updated: 2026-04-05

## PRD-01: Foundation

- [ ] T01.1: Create `src/tag/__init__.py` with version and public API
- [ ] T01.2: Implement Planck radiance function (batched, GPU) in `utils.py`
- [ ] T01.3: Implement cubic B-spline basis constructor in `utils.py`
- [ ] T01.4: Implement second-order difference operator D_beta in `utils.py`
- [ ] T01.5: Define physical constants (h, c, k_B, wavenumber grid)
- [ ] T01.6: Create Pydantic config model + TOML loader
- [ ] T01.7: Write `configs/default.toml`, `configs/paper.toml`, `configs/debug.toml`
- [ ] T01.8: Write `tests/test_model.py` — Planck validation, B-spline shape checks
- [ ] T01.9: Verify Planck output against NIST reference at T=300K, v=1000 cm^-1

## PRD-02: Core Model

- [ ] T02.1: Implement `ThermalForwardModel` (rendering equation S = e*B(T) + (1-e)*X)
- [ ] T02.2: Implement `SLOTDecomposer` with L-BFGS optimizer
- [ ] T02.3: Implement projected gradient for box constraints (0 < e < 1)
- [ ] T02.4: Implement warm-start initialization (T from broadband, beta uniform, V=0.5)
- [ ] T02.5: Implement `HADARDecomposer` baseline (library lookup)
- [ ] T02.6: Batch optimization: reshape [H,W] -> [N] for parallel per-pixel solve
- [ ] T02.7: Test forward model roundtrip: generate S from known T/e/V, verify match
- [ ] T02.8: Test SLOT convergence on 32x32 noise-free synthetic scene

## PRD-03: Loss Functions

- [ ] T03.1: Implement `rendering_residual` (spectral L2 norm)
- [ ] T03.2: Implement `smoothness_penalty` (D_beta * beta L2)
- [ ] T03.3: Implement `emissivity_bound_penalty` (soft barrier)
- [ ] T03.4: Implement `SLOTObjective` combining all terms
- [ ] T03.5: Implement image quality metrics: EN, AG, SF, SD
- [ ] T03.6: Test smoothness penalty is zero for constant beta
- [ ] T03.7: Test objective gradient flows correctly through all terms

## PRD-04: Training Pipeline

- [ ] T04.1: Implement `SyntheticThermalDataset` with configurable scenes
- [ ] T04.2: Implement parametric emissivity models (graybody, Gaussian mixture)
- [ ] T04.3: Implement sensor noise model (NEDT)
- [ ] T04.4: Implement `IRImageDataset` adapter for NUAA-SIRST
- [ ] T04.5: Implement `SLOTRunner` optimization loop with logging
- [ ] T04.6: Implement checkpoint save/load for optimization state
- [ ] T04.7: Implement convergence logging (TensorBoard + JSON)
- [ ] T04.8: Create `scripts/train.py` CLI entry point
- [ ] T04.9: Smoke test: run 5 iterations on debug config, verify checkpoint cycle

## PRD-05: Evaluation

- [ ] T05.1: Implement EN (information entropy) metric
- [ ] T05.2: Implement AG (average gradient) metric
- [ ] T05.3: Implement SF (spatial frequency) metric
- [ ] T05.4: Implement SD (standard deviation) metric
- [ ] T05.5: Implement reconstruction metrics (RMSE_T, MAE_e, SAM)
- [ ] T05.6: Implement SLOT vs HADAR comparison runner
- [ ] T05.7: Create `scripts/evaluate.py` CLI entry point
- [ ] T05.8: Generate evaluation report (markdown + JSON)
- [ ] T05.9: Validate metrics on known reference images

## PRD-06: Export Pipeline

- [ ] T06.1: Export ThermalForwardModel to ONNX
- [ ] T06.2: Build TRT FP16 + FP32 engines via shared toolkit
- [ ] T06.3: Export B-spline basis + parameters as safetensors
- [ ] T06.4: Validate ONNX output matches PyTorch (rtol=1e-5)
- [ ] T06.5: Push exports to HuggingFace

## PRD-07: Integration

- [ ] T07.1: Create `Dockerfile.serve` (3-layer from anima-serve:jazzy)
- [ ] T07.2: Create `docker-compose.serve.yml` with profiles
- [ ] T07.3: Create `.env.serve` with module identity
- [ ] T07.4: Implement `TAGNode(AnimaNode)` in `src/tag/serve.py`
- [ ] T07.5: Implement `/decompose` and `/texture` API endpoints
- [ ] T07.6: Test Docker build + health endpoint
- [ ] T07.7: Test ROS2 node publishes decomposition results
