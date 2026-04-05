# PRD.md — TAG Master Build Plan

> Module: TAG (Thermal Anti-Ghosting)
> Paper: "Universal computational thermal imaging overcoming the ghosting effect" (arXiv 2604.01542)
> Last updated: 2026-04-05

## Overview

TAG is a physics-based computational thermal imaging framework. Unlike most ANIMA modules,
TAG does **not** use neural networks for its core pipeline. The SLOT algorithm performs
constrained optimization using B-spline basis expansion to decompose hyperspectral thermal
data into temperature, emissivity, and texture components.

## Build Plan — 7 PRDs

| PRD | Name | Description | Status | Dependencies |
|-----|------|-------------|--------|-------------|
| PRD-01 | Foundation | Project structure, configs, utils, Planck physics | [x] DONE | None |
| PRD-02 | Core Model + CUDA | SLOT + CUDA kernels (torch.compile, vectorized B-spline) | [x] DONE | PRD-01 |
| PRD-03 | Loss + Datasets | Losses + VIVID++/DroneVehicle thermal adapters | [x] DONE | PRD-01, PRD-02 |
| PRD-04 | Training Pipeline | Multi-dataset CUDA training (synthetic + VIVID++) | [x] DONE | PRD-01, PRD-02, PRD-03 |
| PRD-05 | Evaluation | SLOT vs HADAR comparison, T_RMSE/e_MAE metrics | [x] DONE | PRD-02, PRD-04 |
| PRD-06 | Export Pipeline | pth + safetensors + ONNX + TRT FP16 + TRT FP32 | [x] DONE | PRD-02, PRD-05 |
| PRD-07 | Integration | Docker serving, ROS2 node, API endpoint | [~] INFRA READY | PRD-06 |

## Architecture Summary

```
Input: Hyperspectral thermal cube [H, W, C] (C ~ 67 spectral bands)
  + Sky reference spectrum [C]
  + Ground reference spectrum [C]
      |
      v
SLOT Optimizer (per-pixel or batched):
  - B-spline basis Phi [C, K] (K knots)
  - Temperature T [H, W]
  - Emissivity coefficients beta [H, W, K]
  - View factor V [H, W]
  - Objective: ||S_obs - S_model||^2 + lambda * ||D*beta||^2
  - Constraints: 0 < e < 1
      |
      v
Output:
  - Temperature map [H, W]
  - Emissivity cube [H, W, C]
  - Texture/view-factor map [H, W]
  - High-fidelity grayscale image [H, W]
```

## Key Technical Decisions

1. **GPU acceleration**: Vectorize SLOT across all pixels using PyTorch batched operations
2. **B-spline implementation**: Use `torch` for differentiable B-spline basis evaluation
3. **Optimizer**: L-BFGS with box constraints (scipy.optimize or torch custom)
4. **Synthetic data**: Generate from known T/e/V using Planck's law + noise
5. **Evaluation**: Reproduce Table 1 metrics on synthetic + available IR data
