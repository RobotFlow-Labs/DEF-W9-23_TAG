# PRD-03: Loss Functions

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-01, PRD-02

## Objective

Implement all loss/objective components for the SLOT optimization and evaluation.

## Deliverables

### 1. Rendering Residual Loss (`src/tag/losses.py`)

```python
def rendering_residual(s_obs, s_model):
    """
    L_data = ||S_obs - S_model||^2_2
    Per-pixel spectral L2 norm.
    """
```

### 2. Smoothness Penalty (`src/tag/losses.py`)

```python
def smoothness_penalty(beta, d_beta_matrix):
    """
    L_smooth = (lambda/2) * ||D_beta * beta||^2_2
    Penalizes non-smooth emissivity spectra via second-order differences.
    """
```

### 3. Physics Constraint Loss (`src/tag/losses.py`)

```python
def emissivity_bound_penalty(emissivity, margin=0.01):
    """
    Soft barrier: penalize emissivity outside (0, 1).
    L_bound = sum(relu(-e + margin) + relu(e - 1 + margin))
    """
```

### 4. Combined Objective (`src/tag/losses.py`)

```python
class SLOTObjective(nn.Module):
    """
    Full SLOT objective:
      L = L_data + (lambda/2) * L_smooth + mu * L_bound
    """
```

### 5. Image Quality Losses (for evaluation, not optimization)

```python
def information_entropy(image):
    """EN metric from Table 1."""

def average_gradient(image):
    """AG metric from Table 1."""

def spatial_frequency(image):
    """SF metric from Table 1."""

def standard_deviation(image):
    """SD metric from Table 1."""
```

## Acceptance Criteria
- [ ] Rendering residual is zero for perfect reconstruction
- [ ] Smoothness penalty is zero for constant emissivity
- [ ] Smoothness penalty increases with oscillatory emissivity
- [ ] Combined objective balances data fidelity and smoothness via lambda
- [ ] All image quality metrics match known reference values
