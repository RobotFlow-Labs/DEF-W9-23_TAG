# PRD-02: Core Model — SLOT Algorithm

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-01

## Objective

Implement the SLOT (Smoothness-structured Library-free Optimization for TeX) algorithm
as a PyTorch module that performs per-pixel (batched) TeX decomposition from hyperspectral
thermal input.

## Deliverables

### 1. SLOT Model (`src/tag/model.py`)

```python
class SLOTDecomposer(nn.Module):
    """
    SLOT: Smoothness-structured Library-free Optimization for TeX.

    Given hyperspectral thermal radiance S [B, H, W, C] plus sky/ground references,
    jointly optimizes for:
      - Temperature T [B, H, W]
      - Emissivity coefficients beta [B, H, W, K]
      - View factor V [B, H, W]

    Using: min ||S - S_model||^2 + lambda/2 * ||D*beta||^2
           s.t. 0 < Phi*beta < 1
    """
```

### 2. Forward Model (`src/tag/model.py`)

```python
class ThermalForwardModel(nn.Module):
    """
    Computes modeled spectral radiance from T, e, V:
      S_model = e * B(T) + (1-e) * (V*S_sky + (1-V)*S_ground)
    """
```

### 3. Optimization Strategy

- **Primary**: GPU-accelerated L-BFGS with projected gradient (box constraints)
- **Fallback**: Projected gradient descent with Armijo line search
- Batch across all pixels: reshape [H, W] -> [N] for parallel optimization
- Warm-start: initialize T from broadband mean, beta uniform, V=0.5

### 4. HADAR Baseline (`src/tag/model.py`)

```python
class HADARDecomposer(nn.Module):
    """
    Baseline HADAR method using material library lookup.
    For comparison only.
    """
```

## Acceptance Criteria
- [ ] SLOTDecomposer produces T, emissivity, V from synthetic input
- [ ] Forward model reconstructs radiance with <1% error on noise-free synthetic data
- [ ] Optimization converges in <100 iterations on 32x32 synthetic scene
- [ ] GPU batched: processes 320x256 scene in <30 seconds on L4
- [ ] Box constraints satisfied: all emissivity values in (0, 1)
