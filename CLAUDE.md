# TAG — Thermal Anti-Ghosting (Universal Computational Thermal Imaging)

> Paper: "Universal computational thermal imaging overcoming the ghosting effect"
> arXiv: 2604.01542
> Authors: Hongyi Xu, Du Wang, Chenjun Zhao, Jiashuo Chen, Jiale Lin, Liqin Cao, Yanfei Zhong, Yiyuan She, Fanglin Bao

## Paper Summary

TAG addresses the **ghosting effect** in thermal imaging — loss of texture detail caused
by material non-uniformity and the fundamental degeneracy of thermal radiation (TeX
degeneracy). Multiple combinations of Temperature (T), emissivity (e), and texture/view-
factor (X) produce identical thermal spectra, making unique decomposition impossible
without additional constraints.

The core contribution is **SLOT** (Smoothness-structured Library-free Optimization for
TeX), a nonparametric spectral decomposition method that replaces rigid material-library
lookups (used by HADAR) with cubic B-spline basis expansion + smoothness regularization.
This guarantees a unique globally optimal TeX solution without needing a pre-calibrated
material library.

## Architecture

TAG is **not a neural network**. It is a physics-based computational imaging pipeline:

```
Hyperspectral Thermal Input (320x256 pixels, ~67 spectral bands, 870-1269 cm^-1)
    |
    v
SLOT Optimization (per-pixel B-spline TeX decomposition)
    |-- Emissivity model: e(v) = sum_k beta_k * phi_k(v)  (cubic B-splines)
    |-- Smoothness: lambda/2 * ||D_beta * beta||^2_2
    |-- Physics constraints: 0 < e < 1 (Kirchhoff's law)
    |-- Rendering equation: S = e * B(T) + (1-e) * X
    |-- Ambient: X = V * S_sky + (1-V) * S_ground
    |
    v
Outputs:
    |-- Temperature map T_alpha (per-pixel)
    |-- Spectral emissivity curves e(v) (per-pixel, ~67 channels)
    |-- View factor / texture map V (per-pixel)
    |-- High-fidelity grayscale texture image
```

### SLOT Algorithm Detail

1. **B-spline basis expansion**: e(v) = Phi(v) * beta, where Phi contains K cubic B-spline
   basis functions over wavenumber range [870, 1269] cm^-1
2. **Smoothness penalty**: Second-order difference operator D_beta penalizes curvature:
   [D_beta * beta]_j = beta_j - 2*beta_{j+1} + beta_{j+2}
3. **Regularization parameter**: lambda controls fidelity vs smoothness tradeoff
4. **Optimization**: Constrained optimization with 0 < e < 1 bounds
5. **Guarantee**: Unique globally optimal solution for given lambda

### Comparison with HADAR

| Feature | HADAR | TAG/SLOT |
|---------|-------|----------|
| Material library | Required (pre-calibrated) | Not required |
| Emissivity model | Discrete categories | Continuous B-spline |
| Intra-class variation | Fails (ghosting) | Handles naturally |
| Optimization | Library lookup | Smooth regularization |

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Spectral range | 870-1269 cm^-1 | Long-wave infrared (LWIR) |
| Spectral resolution | 6 cm^-1 | ~67 spectral bands |
| Spatial resolution | 320 x 256 pixels | MCT focal plane array |
| B-spline order | Cubic (order 4) | Standard cubic B-splines |
| Number of knots K | ~20-30 (tunable) | Paper tests varying K |
| Regularization lambda | 1e-3 to 1e6 | Continuously tunable |
| Emissivity bounds | (0, 1) | Physics constraint |

## Datasets

| Dataset | Description | Use |
|---------|-------------|-----|
| Custom facial thermal | Hyperspectral thermal of human faces (Wuhan) | Primary evaluation |
| DARPA Invisible Headlights | Military thermal scenes | Robustness (no material library) |

**Note**: This paper uses **custom hardware data** (Hypercam-LW, Telops Inc.), not
standard vision benchmarks. For ANIMA reproduction, we synthesize hyperspectral thermal
data from available infrared datasets or use publicly available LWIR hyperspectral data.

## Evaluation Metrics

### Low-level Image Quality (Table 1)
- **EN** (Information Entropy) - higher is better
- **AG** (Average Gradient) - higher is better
- **SF** (Spatial Frequency) - higher is better
- **SD** (Standard Deviation) - higher is better

### High-level Downstream Tasks
- **Task I**: AI colorization quality (thermal-to-RGB)
- **Task II**: 3D topological alignment (MediaPipe 468-point facial mesh)
- **Task III**: Semantic emotion recognition (MTCNN + FER)

## Downstream Models (inference only, not trained)
- Google MediaPipe — 468-point facial mesh
- MTCNN — face detection
- FER (Facial Expression Recognition) — emotion classification

## Key Equations

### Rendering equation
```
S_{alpha,v} = e_{alpha,v} * B_v(T_alpha) + (1 - e_{alpha,v}) * X_{alpha,v}
```

### Planck's law
```
B_v(T) = (2*h*v^3/c^2) / (exp(h*v/(k_B*T)) - 1)
```

### Ambient texture
```
X = V * S_sky + (1-V) * S_ground
```

### SLOT objective
```
min_{T, beta, V}  ||S_obs - S_model(T, Phi*beta, V)||^2 + (lambda/2) * ||D_beta * beta||^2
subject to  0 < Phi(v)*beta < 1  for all v
```
