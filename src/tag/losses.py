"""Loss functions for TAG SLOT optimization and evaluation metrics.

The SLOT objective combines:
  L = L_data + (lambda/2) * L_smooth + mu * L_bound

Image quality metrics reproduce Table 1 from the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from tag.utils import image_gradient

# ---------------------------------------------------------------------------
# SLOT Optimization Losses
# ---------------------------------------------------------------------------


def rendering_residual(s_obs: Tensor, s_model: Tensor) -> Tensor:
    """Spectral rendering residual: per-pixel L2 norm.

    L_data = mean_pixels( ||S_obs - S_model||^2_2 )

    Args:
        s_obs: [N, C] observed spectral radiance.
        s_model: [N, C] modeled spectral radiance.

    Returns:
        Scalar mean squared residual.
    """
    return ((s_obs - s_model) ** 2).sum(dim=-1).mean()


def smoothness_penalty(beta: Tensor, d_beta: Tensor) -> Tensor:
    """Emissivity smoothness penalty via second-order differences.

    L_smooth = mean_pixels( ||D_beta * beta||^2_2 )

    Args:
        beta: [N, K] B-spline coefficients.
        d_beta: [K-2, K] second-order difference operator.

    Returns:
        Scalar smoothness penalty.
    """
    # [N, K] @ [K, K-2] = [N, K-2]
    diff = beta @ d_beta.T
    return (diff**2).sum(dim=-1).mean()


def emissivity_bound_penalty(emissivity: Tensor, margin: float = 0.01) -> Tensor:
    """Soft barrier penalty for emissivity bounds (0, 1).

    L_bound = mean( relu(-e + margin) + relu(e - 1 + margin) )

    Args:
        emissivity: [N, C] spectral emissivity values.
        margin: safety margin from bounds (default 0.01).

    Returns:
        Scalar bound violation penalty.
    """
    lower_violation = torch.relu(-emissivity + margin)
    upper_violation = torch.relu(emissivity - 1.0 + margin)
    return (lower_violation + upper_violation).sum(dim=-1).mean()


class SLOTObjective(nn.Module):
    """Combined SLOT objective function.

    L = L_data + (lambda/2) * L_smooth + mu * L_bound

    Args:
        reg_lambda: smoothness regularization weight.
        constraint_mu: emissivity bound penalty weight.
        bound_margin: emissivity bound margin.
    """

    def __init__(
        self,
        reg_lambda: float = 1.0,
        constraint_mu: float = 10.0,
        bound_margin: float = 0.01,
    ):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.constraint_mu = constraint_mu
        self.bound_margin = bound_margin

    def forward(
        self,
        s_obs: Tensor,
        s_model: Tensor,
        beta: Tensor,
        d_beta: Tensor,
        emissivity: Tensor,
    ) -> dict[str, Tensor]:
        """Compute SLOT objective with breakdown.

        Args:
            s_obs: [N, C] observed radiance.
            s_model: [N, C] modeled radiance.
            beta: [N, K] B-spline coefficients.
            d_beta: [K-2, K] difference operator.
            emissivity: [N, C] spectral emissivity.

        Returns:
            Dict with 'total', 'residual', 'smoothness', 'bound' loss values.
        """
        l_data = rendering_residual(s_obs, s_model)
        l_smooth = smoothness_penalty(beta, d_beta)
        l_bound = emissivity_bound_penalty(emissivity, self.bound_margin)

        total = l_data + (self.reg_lambda / 2.0) * l_smooth + self.constraint_mu * l_bound

        return {
            "total": total,
            "residual": l_data,
            "smoothness": l_smooth,
            "bound": l_bound,
        }


# ---------------------------------------------------------------------------
# Image Quality Metrics (Table 1 from paper)
# ---------------------------------------------------------------------------


def information_entropy(image: Tensor, n_bins: int = 256) -> Tensor:
    """Information Entropy (EN) of a grayscale image.

    EN = -sum(p * log2(p)) over histogram bins.

    Args:
        image: [H, W] grayscale image (normalized to [0, 1] or [0, 255]).

    Returns:
        Scalar entropy value.
    """
    # Normalize to [0, n_bins-1]
    img = image.detach().float()
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img = (img - img_min) / (img_max - img_min) * (n_bins - 1)
    img = img.long().clamp(0, n_bins - 1)

    # Histogram
    hist = torch.zeros(n_bins, device=image.device)
    hist.scatter_add_(0, img.flatten(), torch.ones_like(img.flatten(), dtype=torch.float))

    # Probability
    p = hist / hist.sum()
    p = p[p > 0]

    return -(p * torch.log2(p)).sum()


def average_gradient(image: Tensor) -> Tensor:
    """Average Gradient (AG) of a grayscale image.

    AG = mean( sqrt(dx^2 + dy^2) / 2 )

    Args:
        image: [H, W] grayscale image.

    Returns:
        Scalar AG value.
    """
    dx, dy = image_gradient(image)
    grad_mag = torch.sqrt(dx**2 + dy**2) / 2.0
    return grad_mag.mean()


def spatial_frequency(image: Tensor) -> Tensor:
    """Spatial Frequency (SF) of a grayscale image.

    SF = sqrt(RF^2 + CF^2)
    where RF = sqrt(mean(row_diff^2)), CF = sqrt(mean(col_diff^2))

    Args:
        image: [H, W] grayscale image.

    Returns:
        Scalar SF value.
    """
    # Row frequency: differences along columns (horizontal)
    row_diff = image[:, 1:] - image[:, :-1]
    rf = torch.sqrt((row_diff**2).mean())

    # Column frequency: differences along rows (vertical)
    col_diff = image[1:, :] - image[:-1, :]
    cf = torch.sqrt((col_diff**2).mean())

    return torch.sqrt(rf**2 + cf**2)


def standard_deviation(image: Tensor) -> Tensor:
    """Standard Deviation (SD) of pixel values.

    Args:
        image: [H, W] grayscale image.

    Returns:
        Scalar SD value.
    """
    return image.float().std()


def spectral_angle_mapper(pred: Tensor, target: Tensor) -> Tensor:
    """Spectral Angle Mapper (SAM) between predicted and target spectra.

    SAM = arccos( dot(pred, target) / (||pred|| * ||target||) )
    Averaged over all pixels.

    Args:
        pred: [N, C] predicted spectra.
        target: [N, C] target spectra.

    Returns:
        Scalar mean SAM in radians.
    """
    dot = (pred * target).sum(dim=-1)
    norm_pred = pred.norm(dim=-1)
    norm_target = target.norm(dim=-1)

    cos_angle = dot / (norm_pred * norm_target + 1e-8)
    cos_angle = cos_angle.clamp(-1.0, 1.0)

    return torch.arccos(cos_angle).mean()
