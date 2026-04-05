"""Physics utilities, B-spline basis, and constants for TAG.

All wavenumber values are in cm^-1. Radiance is in W/(m^2 sr cm^-1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as f_nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Physical constants (SI, but wavenumber in cm^-1)
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-34       # J*s
C_LIGHT = 2.99792458e8          # m/s
K_BOLTZMANN = 1.380649e-23      # J/K
C_LIGHT_CM = C_LIGHT * 100.0    # cm/s  (for wavenumber formulas)

# First and second radiation constants for wavenumber form
C1 = 2.0 * H_PLANCK * C_LIGHT_CM**2   # W*cm^2/sr
C2 = H_PLANCK * C_LIGHT_CM / K_BOLTZMANN  # cm*K

# Default spectral grid matching the paper's Hypercam-LW settings
WAVENUM_MIN = 870.0    # cm^-1
WAVENUM_MAX = 1269.0   # cm^-1
WAVENUM_STEP = 6.0     # cm^-1


def default_wavenumber_grid(device: torch.device | None = None) -> Tensor:
    """Return the default wavenumber grid [C] matching paper (870-1269 cm^-1, 6 cm^-1)."""
    grid = torch.arange(WAVENUM_MIN, WAVENUM_MAX + WAVENUM_STEP / 2, WAVENUM_STEP)
    if device is not None:
        grid = grid.to(device)
    return grid


# ---------------------------------------------------------------------------
# Planck's law (wavenumber form)
# ---------------------------------------------------------------------------

def planck_radiance(wavenumber: Tensor, temperature: Tensor) -> Tensor:
    """Compute Planck spectral radiance B_v(T) in wavenumber space.

    B_v(v, T) = C1 * v^3 / (exp(C2 * v / T) - 1)

    Args:
        wavenumber: [C] wavenumber values in cm^-1.
        temperature: [...] temperature values in Kelvin (broadcastable).

    Returns:
        Radiance [..., C] in W/(m^2 sr cm^-1).
    """
    # Expand for broadcasting: wavenumber [C], temperature [..., 1]
    v = wavenumber  # [C]
    t_expanded = temperature.unsqueeze(-1)  # [..., 1]

    # Avoid overflow in exp by clamping exponent
    exponent = C2 * v / t_expanded  # [..., C]
    exponent = torch.clamp(exponent, max=500.0)

    radiance = C1 * v.pow(3) / (torch.exp(exponent) - 1.0)
    return radiance  # [..., C]


# ---------------------------------------------------------------------------
# Cubic B-spline basis
# ---------------------------------------------------------------------------

def _bspline_basis_single(x: Tensor, knots: Tensor, i: int, order: int) -> Tensor:
    """Evaluate single B-spline basis function using Cox-de Boor recursion.

    Args:
        x: [M] evaluation points.
        knots: [n_knots + order] knot vector.
        i: basis function index.
        order: spline order (4 for cubic).

    Returns:
        [M] basis function values.
    """
    if order == 1:
        return ((x >= knots[i]) & (x < knots[i + 1])).float()

    denom1 = knots[i + order - 1] - knots[i]
    denom2 = knots[i + order] - knots[i + 1]

    term1 = torch.zeros_like(x)
    term2 = torch.zeros_like(x)

    if denom1.abs() > 1e-12:
        term1 = (x - knots[i]) / denom1 * _bspline_basis_single(
            x, knots, i, order - 1
        )
    if denom2.abs() > 1e-12:
        term2 = (knots[i + order] - x) / denom2 * _bspline_basis_single(
            x, knots, i + 1, order - 1
        )

    return term1 + term2


def cubic_bspline_basis(
    wavenumber_grid: Tensor,
    n_knots: int,
    order: int = 4,
) -> Tensor:
    """Construct cubic B-spline basis matrix Phi.

    Args:
        wavenumber_grid: [C] wavenumber evaluation points.
        n_knots: number of interior knots (K = n_knots + order - 2 basis functions).
        order: spline order (default 4 = cubic).

    Returns:
        Phi: [C, K] basis matrix where K = n_knots + order - 2.
    """
    device = wavenumber_grid.device
    v_min = wavenumber_grid.min().item()
    v_max = wavenumber_grid.max().item()

    # Create uniform knot vector with clamped ends
    interior = torch.linspace(v_min, v_max, n_knots, device=device)
    pad_left = torch.full((order - 1,), v_min, device=device)
    pad_right = torch.full((order - 1,), v_max, device=device)
    knot_vector = torch.cat([pad_left, interior, pad_right])

    n_basis = len(knot_vector) - order  # K

    # Evaluate each basis function at all grid points
    phi_cols = []
    for i in range(n_basis):
        phi_i = _bspline_basis_single(wavenumber_grid, knot_vector, i, order)
        phi_cols.append(phi_i)

    phi = torch.stack(phi_cols, dim=-1)  # [C, K]

    # Fix last point (B-spline convention: rightmost point is in last basis)
    phi[-1, -1] = 1.0

    return phi


def second_order_diff_operator(n_basis: int, device: torch.device | None = None) -> Tensor:
    """Construct second-order difference operator D_beta.

    [D * beta]_j = beta_j - 2*beta_{j+1} + beta_{j+2}

    Args:
        n_basis: number of B-spline basis functions K.
        device: target device.

    Returns:
        D: [K-2, K] second-order difference matrix.
    """
    n_rows = n_basis - 2
    d = torch.zeros(n_rows, n_basis, device=device)
    for j in range(n_rows):
        d[j, j] = 1.0
        d[j, j + 1] = -2.0
        d[j, j + 2] = 1.0
    return d


# ---------------------------------------------------------------------------
# Ambient texture model
# ---------------------------------------------------------------------------

def ambient_texture(
    view_factor: Tensor,
    s_sky: Tensor,
    s_ground: Tensor,
) -> Tensor:
    """Compute ambient texture X = V * S_sky + (1-V) * S_ground.

    Args:
        view_factor: [...] view factor V in [0, 1].
        s_sky: [C] sky radiance spectrum.
        s_ground: [C] ground radiance spectrum.

    Returns:
        X: [..., C] ambient texture spectrum.
    """
    v = view_factor.unsqueeze(-1)  # [..., 1]
    return v * s_sky + (1.0 - v) * s_ground


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def image_gradient(img: Tensor) -> tuple[Tensor, Tensor]:
    """Compute spatial gradients (Sobel-like finite differences).

    Args:
        img: [B, 1, H, W] or [H, W] grayscale image.

    Returns:
        (dx, dy): horizontal and vertical gradients, same shape as input.
    """
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Simple finite difference
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]

    # Pad to original size
    dx = f_nn.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = f_nn.pad(dy, (0, 0, 0, 1), mode="replicate")

    if squeeze:
        dx = dx.squeeze(0).squeeze(0)
        dy = dy.squeeze(0).squeeze(0)

    return dx, dy


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
