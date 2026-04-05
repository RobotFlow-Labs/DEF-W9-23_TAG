# PRD-01: Foundation

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)

## Objective

Set up project structure, physics utilities (Planck's law, B-spline basis), configuration
system, and basic data structures for hyperspectral thermal processing.

## Deliverables

### 1. Project Structure
- `src/tag/__init__.py` — package init with version
- `src/tag/utils.py` — physics utilities, B-spline helpers, constants
- `src/tag/types.py` — typed data structures (TexResult, HyperspectralCube, etc.)

### 2. Physics Constants & Functions
- Planck's law: `planck_radiance(wavenumber, temperature)` — batched GPU
- Kirchhoff's law: reflectance = 1 - emissivity
- Wavenumber grid: 870-1269 cm^-1 at 6 cm^-1 resolution (~67 bands)
- Physical constants: h, c, k_B in CGS/SI units

### 3. B-Spline Basis
- `cubic_bspline_basis(wavenumber_grid, n_knots)` — returns Phi matrix [C, K]
- `second_order_diff_operator(n_knots)` — returns D_beta [K-2, K]
- Differentiable via PyTorch autograd

### 4. Configuration
- `configs/default.toml` — base config (spectral range, resolution, lambda, K)
- `configs/paper.toml` — paper-exact parameters
- `configs/debug.toml` — small-scale testing (tiny spatial, few bands)
- Pydantic BaseSettings model for config loading

### 5. Tests
- `tests/test_model.py` — Planck law output shape, B-spline basis properties
- Verify Planck matches reference values at known T

## Acceptance Criteria
- [ ] Planck radiance produces correct values for T=300K at 1000 cm^-1
- [ ] B-spline basis matrix has correct shape [C, K] and is differentiable
- [ ] D_beta operator produces correct second-order differences
- [ ] Config loads from TOML with Pydantic validation
- [ ] All tests pass with `uv run pytest tests/ -x -v`
