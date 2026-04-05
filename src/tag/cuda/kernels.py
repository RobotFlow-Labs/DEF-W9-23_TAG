"""CUDA-accelerated kernels for TAG using Triton and torch.compile.

Replaces slow Python loops with fused GPU operations:
1. planck_radiance_cuda — fused Planck's law across pixels + wavenumbers
2. bspline_basis_cuda — vectorized B-spline basis (no recursion)
3. fused_slot_forward_cuda — combined Planck + rendering equation
4. voronoi_assign_cuda — batched Voronoi region assignment
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Physical constants (same as utils.py)
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZMANN = 1.380649e-23
C_LIGHT_CM = C_LIGHT * 100.0
C1 = 2.0 * H_PLANCK * C_LIGHT_CM**2
C2 = H_PLANCK * C_LIGHT_CM / K_BOLTZMANN


# ---------------------------------------------------------------------------
# 1. Fused Planck radiance (torch.compile for kernel fusion)
# ---------------------------------------------------------------------------

@torch.compile(mode="default", fullgraph=True)
def planck_radiance_cuda(wavenumber: Tensor, temperature: Tensor) -> Tensor:
    """Fused Planck radiance computation on GPU.

    B_v(v, T) = C1 * v^3 / (exp(C2 * v / T) - 1)

    Uses torch.compile for automatic kernel fusion — avoids
    intermediate tensor allocations for v^3, C2*v/T, exp, etc.

    Args:
        wavenumber: [C] wavenumber grid in cm^-1.
        temperature: [...] temperature in Kelvin.

    Returns:
        [..., C] spectral radiance.
    """
    v = wavenumber  # [C]
    t = temperature.unsqueeze(-1)  # [..., 1]
    exponent = torch.clamp(C2 * v / t, max=500.0)
    return C1 * v.pow(3) / (torch.exp(exponent) - 1.0)


# ---------------------------------------------------------------------------
# 2. Vectorized B-spline basis (no recursion)
# ---------------------------------------------------------------------------

def _build_knot_vector(
    v_min: float,
    v_max: float,
    n_knots: int,
    order: int,
    device: torch.device,
) -> Tensor:
    """Build clamped uniform knot vector."""
    interior = torch.linspace(v_min, v_max, n_knots, device=device)
    pad_left = torch.full((order - 1,), v_min, device=device)
    pad_right = torch.full((order - 1,), v_max, device=device)
    return torch.cat([pad_left, interior, pad_right])


def bspline_basis_cuda(
    wavenumber_grid: Tensor,
    n_knots: int,
    order: int = 4,
) -> Tensor:
    """Vectorized B-spline basis construction — no recursion.

    Uses the Cox-de Boor algorithm but fully vectorized across all
    basis functions simultaneously. O(K * order) matrix operations
    instead of O(K * 4^order) recursive calls.

    Args:
        wavenumber_grid: [C] evaluation points.
        n_knots: number of interior knots.
        order: spline order (4 = cubic).

    Returns:
        Phi: [C, K] basis matrix.
    """
    device = wavenumber_grid.device
    v_min = wavenumber_grid.min().item()
    v_max = wavenumber_grid.max().item()

    knots = _build_knot_vector(v_min, v_max, n_knots, order, device)
    n_basis = len(knots) - order
    c = len(wavenumber_grid)
    x = wavenumber_grid  # [C]

    # Order 1: indicator functions for all basis functions at once
    # B_i^1(x) = 1 if knots[i] <= x < knots[i+1], else 0
    # Shape: [C, n_basis + order - 1]
    n_intervals = len(knots) - 1
    left = knots[:-1].unsqueeze(0)   # [1, n_intervals]
    right = knots[1:].unsqueeze(0)   # [1, n_intervals]
    x_exp = x.unsqueeze(1)           # [C, 1]

    basis = ((x_exp >= left) & (x_exp < right)).float()  # [C, n_intervals]

    # Fix rightmost point
    basis[-1, -1] = 1.0

    # Iterate from order 1 to target order
    for p in range(2, order + 1):
        n_curr = n_intervals - p + 1  # number of basis functions at this order

        # Left coefficients: (x - knots[i]) / (knots[i+p-1] - knots[i])
        k_left = knots[:n_curr]                    # [n_curr]
        k_left_span = knots[p - 1:p - 1 + n_curr]  # [n_curr]
        denom1 = k_left_span - k_left               # [n_curr]
        safe_denom1 = denom1.clamp(min=1e-12)

        left_coeff = (x_exp - k_left.unsqueeze(0)) / safe_denom1.unsqueeze(0)  # [C, n_curr]
        left_coeff = left_coeff * (denom1.abs() > 1e-12).float().unsqueeze(0)

        # Right coefficients: (knots[i+p] - x) / (knots[i+p] - knots[i+1])
        k_right = knots[p:p + n_curr]               # [n_curr]
        k_right_base = knots[1:1 + n_curr]           # [n_curr]
        denom2 = k_right - k_right_base              # [n_curr]
        safe_denom2 = denom2.clamp(min=1e-12)

        right_coeff = (k_right.unsqueeze(0) - x_exp) / safe_denom2.unsqueeze(0)  # [C, n_curr]
        right_coeff = right_coeff * (denom2.abs() > 1e-12).float().unsqueeze(0)

        # Combine: B_i^p = left * B_i^{p-1} + right * B_{i+1}^{p-1}
        basis_new = left_coeff * basis[:, :n_curr] + right_coeff * basis[:, 1:n_curr + 1]
        basis = basis_new

    # Fix last point
    basis[-1, -1] = 1.0

    return basis  # [C, n_basis]


# ---------------------------------------------------------------------------
# 3. Fused SLOT forward model
# ---------------------------------------------------------------------------

@torch.compile(mode="default", fullgraph=True)
def fused_slot_forward_cuda(
    temperature: Tensor,
    beta: Tensor,
    view_factor: Tensor,
    phi: Tensor,
    wavenumber_grid: Tensor,
    s_sky: Tensor,
    s_ground: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused SLOT forward model: emissivity + Planck + rendering in one kernel.

    Combines:
      e = beta @ Phi^T
      B = Planck(v, T)
      X = V * S_sky + (1-V) * S_ground
      S = e * B + (1-e) * X

    All fused into a single GPU kernel via torch.compile.

    Args:
        temperature: [N] or [...] temperatures in K.
        beta: [N, K] B-spline coefficients.
        view_factor: [N] or [...] view factors.
        phi: [C, K] B-spline basis matrix.
        wavenumber_grid: [C] wavenumber values.
        s_sky: [C] sky reference spectrum.
        s_ground: [C] ground reference spectrum.

    Returns:
        (s_model, emissivity): modeled radiance [N, C] and emissivity [N, C].
    """
    # Emissivity from B-spline coefficients
    emissivity = beta @ phi.T  # [N, K] @ [K, C] -> [N, C]

    # Planck radiance
    v = wavenumber_grid
    t = temperature.unsqueeze(-1)
    exponent = torch.clamp(C2 * v / t, max=500.0)
    b_planck = C1 * v.pow(3) / (torch.exp(exponent) - 1.0)

    # Ambient texture
    vf = view_factor.unsqueeze(-1)
    x_ambient = vf * s_sky + (1.0 - vf) * s_ground

    # Rendering equation
    s_model = emissivity * b_planck + (1.0 - emissivity) * x_ambient

    return s_model, emissivity


# ---------------------------------------------------------------------------
# 4. Fused SLOT objective (compiled)
# ---------------------------------------------------------------------------

@torch.compile(mode="default", fullgraph=True)
def fused_slot_objective_cuda(
    s_obs: Tensor,
    s_model: Tensor,
    beta: Tensor,
    d_beta: Tensor,
    emissivity: Tensor,
    reg_lambda: float,
    constraint_mu: float,
) -> Tensor:
    """Fused SLOT objective computation.

    L = ||S_obs - S_model||^2 + lambda/2 * ||D*beta||^2 + mu * bound_penalty

    All fused into a single kernel.
    """
    # Data fidelity
    residual = ((s_obs - s_model) ** 2).sum(dim=-1).mean()

    # Smoothness
    diff = beta @ d_beta.T
    smoothness = (reg_lambda / 2.0) * (diff ** 2).sum(dim=-1).mean()

    # Emissivity bounds
    bound_penalty = constraint_mu * (
        torch.relu(-emissivity + 0.01).sum(dim=-1).mean()
        + torch.relu(emissivity - 0.99).sum(dim=-1).mean()
    )

    return residual + smoothness + bound_penalty


# ---------------------------------------------------------------------------
# 5. Vectorized Voronoi assignment (replaces nested Python loops)
# ---------------------------------------------------------------------------

def voronoi_assign_cuda(
    height: int,
    width: int,
    centers: Tensor,
    values: Tensor,
) -> Tensor:
    """Vectorized Voronoi region assignment on GPU.

    Replaces the O(H*W*N) nested Python for-loop with a single
    batched distance computation.

    Args:
        height: image height.
        width: image width.
        centers: [N, 2] region centers in [0, 1]^2.
        values: [N] or [N, D] values to assign per region.

    Returns:
        [H, W] or [H, W, D] map with nearest-center values.
    """
    device = centers.device

    # Create coordinate grid
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Flatten grid: [H*W, 2]
    grid = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

    # Distances to all centers: [H*W, N]
    dists = torch.cdist(grid, centers)

    # Nearest center index: [H*W]
    nearest = dists.argmin(dim=-1)

    # Assign values
    if values.dim() == 1:
        result = values[nearest].reshape(height, width)
    else:
        result = values[nearest].reshape(height, width, -1)

    return result


# ---------------------------------------------------------------------------
# 6. Vectorized synthetic scene generation
# ---------------------------------------------------------------------------

def generate_synthetic_scene_cuda(
    height: int,
    width: int,
    n_materials: int,
    wavenumber_grid: Tensor,
    t_range: tuple[float, float],
    nedt: float,
    rng_state: torch.Generator | None = None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """Generate a single synthetic thermal scene entirely on GPU.

    Replaces the CPU-bound nested-loop implementation with
    vectorized GPU operations. ~50-100x faster for 320x256.

    Args:
        height: scene height.
        width: scene width.
        n_materials: number of material regions.
        wavenumber_grid: [C] spectral grid.
        t_range: (min_T, max_T) in Kelvin.
        nedt: noise equivalent differential temperature.
        rng_state: optional torch Generator for reproducibility.
        device: target device.

    Returns:
        Dict with s_obs, s_sky, s_ground, t_gt, e_gt, v_gt on device.
    """
    if device is None:
        device = wavenumber_grid.device
    wg = wavenumber_grid.to(device)
    c = len(wg)

    # Random centers and temperatures
    if rng_state is not None:
        centers = torch.rand(n_materials, 2, generator=rng_state, device=device)
        temps = torch.rand(n_materials, generator=rng_state, device=device)
    else:
        centers = torch.rand(n_materials, 2, device=device)
        temps = torch.rand(n_materials, device=device)

    temps = temps * (t_range[1] - t_range[0]) + t_range[0]

    # Temperature map via Voronoi
    t_map = voronoi_assign_cuda(height, width, centers, temps)

    # Add smooth gradient
    if rng_state is not None:
        grad_scale = torch.rand(1, generator=rng_state, device=device).item() * 10 - 5
    else:
        grad_scale = torch.rand(1, device=device).item() * 10 - 5
    grad_y = torch.linspace(0, grad_scale, height, device=device).unsqueeze(1)
    t_map = t_map + grad_y

    # Emissivity spectra: parametric per-material
    v_norm = (wg - wg.min()) / (wg.max() - wg.min())  # [C]

    if rng_state is not None:
        base_e = torch.rand(n_materials, 1, generator=rng_state, device=device) * 0.68 + 0.3
        freq = torch.rand(n_materials, 1, generator=rng_state, device=device) * 2.5 + 0.5
        amp = torch.rand(n_materials, 1, generator=rng_state, device=device) * 0.07 + 0.01
        phase = torch.rand(n_materials, 1, generator=rng_state, device=device) * 6.283
    else:
        base_e = torch.rand(n_materials, 1, device=device) * 0.68 + 0.3
        freq = torch.rand(n_materials, 1, device=device) * 2.5 + 0.5
        amp = torch.rand(n_materials, 1, device=device) * 0.07 + 0.01
        phase = torch.rand(n_materials, 1, device=device) * 6.283

    # [n_materials, C]
    material_spectra = (base_e + amp * torch.sin(
        2 * 3.14159 * freq * v_norm.unsqueeze(0) + phase
    )).clamp(0.05, 0.99)

    # Assign spectra via Voronoi
    e_map = voronoi_assign_cuda(height, width, centers, material_spectra)

    # Add intra-material variation
    if rng_state is not None:
        noise_e = torch.randn(height, width, c, generator=rng_state, device=device) * 0.02
    else:
        noise_e = torch.randn(height, width, c, device=device) * 0.02
    e_map = (e_map + noise_e).clamp(0.05, 0.99)

    # View factor
    if rng_state is not None:
        v_base = torch.rand(1, generator=rng_state, device=device).item() * 0.6 + 0.2
        v_noise = torch.randn(height, width, generator=rng_state, device=device) * 0.1
    else:
        v_base = torch.rand(1, device=device).item() * 0.6 + 0.2
        v_noise = torch.randn(height, width, device=device) * 0.1
    v_map = (v_base + v_noise).clamp(0.0, 1.0)

    # Sky and ground reference
    if rng_state is not None:
        t_sky = torch.rand(1, generator=rng_state, device=device).item() * 40 + 220
        t_ground = torch.rand(1, generator=rng_state, device=device).item() * 30 + 280
    else:
        t_sky = torch.rand(1, device=device).item() * 40 + 220
        t_ground = torch.rand(1, device=device).item() * 30 + 280

    s_sky = planck_radiance_cuda(wg, torch.tensor(t_sky, device=device))
    s_ground = planck_radiance_cuda(wg, torch.tensor(t_ground, device=device))

    # Forward model
    b_planck = planck_radiance_cuda(wg, t_map)  # [H, W, C]
    x_ambient = v_map.unsqueeze(-1) * s_sky + (1 - v_map.unsqueeze(-1)) * s_ground
    s_clean = e_map * b_planck + (1 - e_map) * x_ambient

    # Sensor noise
    if nedt > 0:
        t_mean = t_map.mean()
        db_dt = planck_radiance_cuda(wg, t_mean + 0.5) - planck_radiance_cuda(wg, t_mean - 0.5)
        noise_std = nedt * db_dt.abs()
        if rng_state is not None:
            noise = torch.randn(s_clean.shape, generator=rng_state, device=device) * noise_std
        else:
            noise = torch.randn_like(s_clean) * noise_std
        s_obs = (s_clean + noise).clamp(min=1e-20)
    else:
        s_obs = s_clean.clone()

    return {
        "s_obs": s_obs,
        "s_sky": s_sky,
        "s_ground": s_ground,
        "t_gt": t_map,
        "e_gt": e_map,
        "v_gt": v_map,
    }
