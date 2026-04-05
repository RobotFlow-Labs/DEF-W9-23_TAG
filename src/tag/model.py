"""Core TAG models: SLOT decomposer, thermal forward model, HADAR baseline.

The SLOT algorithm solves the TeX decomposition problem:
  Given observed spectral radiance S_obs [N, C],
  recover Temperature T [N], emissivity e [N, C], and view factor V [N]
  by minimizing ||S_obs - S_model||^2 + lambda/2 * ||D*beta||^2
  subject to 0 < e < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from tag.utils import (
    ambient_texture,
    cubic_bspline_basis,
    default_wavenumber_grid,
    planck_radiance,
    second_order_diff_operator,
)


@dataclass
class TexResult:
    """Result of TeX decomposition."""

    temperature: Tensor      # [B, H, W] or [N] in Kelvin
    emissivity: Tensor       # [B, H, W, C] or [N, C]
    view_factor: Tensor      # [B, H, W] or [N]
    beta: Tensor             # [B, H, W, K] or [N, K] B-spline coefficients
    texture: Tensor          # [B, H, W] or [N] high-fidelity grayscale
    s_reconstructed: Tensor  # [B, H, W, C] or [N, C] reconstructed radiance
    objective: float         # final objective value
    n_iterations: int        # iterations used


class ThermalForwardModel(nn.Module):
    """Compute modeled spectral radiance from T, e, V.

    S_model = e * B(T) + (1-e) * X
    where X = V * S_sky + (1-V) * S_ground
    """

    def __init__(self, wavenumber_grid: Tensor | None = None):
        super().__init__()
        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.register_buffer("wavenumber_grid", wavenumber_grid)

    def forward(
        self,
        temperature: Tensor,
        emissivity: Tensor,
        view_factor: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> Tensor:
        """Compute modeled spectral radiance.

        Args:
            temperature: [...] temperature in Kelvin.
            emissivity: [..., C] spectral emissivity in (0, 1).
            view_factor: [...] view factor in [0, 1].
            s_sky: [C] sky reference spectrum.
            s_ground: [C] ground reference spectrum.

        Returns:
            S_model: [..., C] modeled spectral radiance.
        """
        # Planck radiance: [..., C]
        b_planck = planck_radiance(self.wavenumber_grid, temperature)

        # Ambient texture: [..., C]
        x_ambient = ambient_texture(view_factor, s_sky, s_ground)

        # Rendering equation
        s_model = emissivity * b_planck + (1.0 - emissivity) * x_ambient

        return s_model


class SLOTDecomposer(nn.Module):
    """SLOT: Smoothness-structured Library-free Optimization for TeX.

    Performs per-pixel (batched) constrained optimization to decompose
    hyperspectral thermal radiance into T, e(v), and V.
    """

    def __init__(
        self,
        n_knots: int = 20,
        reg_lambda: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        lr: float = 0.01,
        constraint_mu: float = 10.0,
        wavenumber_grid: Tensor | None = None,
    ):
        """Initialize SLOT decomposer.

        Args:
            n_knots: number of interior B-spline knots.
            reg_lambda: smoothness regularization weight.
            max_iter: maximum optimization iterations.
            tolerance: convergence tolerance on objective change.
            lr: learning rate for L-BFGS.
            constraint_mu: penalty weight for emissivity bound violations.
            wavenumber_grid: [C] wavenumber grid (default: paper settings).
        """
        super().__init__()
        self.n_knots = n_knots
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lr = lr
        self.constraint_mu = constraint_mu

        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.register_buffer("wavenumber_grid", wavenumber_grid)

        # Build B-spline basis and difference operator
        phi = cubic_bspline_basis(wavenumber_grid, n_knots)
        d_beta = second_order_diff_operator(phi.shape[1])
        self.register_buffer("phi", phi)       # [C, K]
        self.register_buffer("d_beta", d_beta) # [K-2, K]

        self.forward_model = ThermalForwardModel(wavenumber_grid)
        self.n_basis = phi.shape[1]

    def _initialize(
        self,
        s_obs: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Warm-start initialization for optimization variables.

        Args:
            s_obs: [N, C] observed spectral radiance.
            s_sky: [C] sky reference.
            s_ground: [C] ground reference.

        Returns:
            (T_init, beta_init, V_init) initial values.
        """
        n_pixels = s_obs.shape[0]
        device = s_obs.device

        # Temperature: estimate from broadband mean radiance
        # Invert Planck at mean wavenumber for broadband average
        mean_radiance = s_obs.mean(dim=-1)  # [N]
        v_mean = self.wavenumber_grid.mean()

        # Rough inversion: T ~ C2 * v_mean / ln(C1 * v_mean^3 / S_mean + 1)
        from tag.utils import C1, C2
        numer = C1 * v_mean.pow(3)
        ratio = numer / (mean_radiance + 1e-20)
        t_init = C2 * v_mean / torch.log(ratio + 1.0)
        t_init = t_init.clamp(200.0, 500.0)  # physical range

        # Beta: uniform emissivity ~0.9 -> solve for beta via least-squares
        # Phi * beta = 0.9 -> beta = pinv(Phi) * 0.9
        target_e = torch.full((n_pixels, self.phi.shape[0]), 0.9, device=device)
        phi_pinv = torch.linalg.pinv(self.phi)  # [K, C]
        beta_init = (target_e @ phi_pinv.T)  # [N, K]

        # View factor: start at 0.5
        v_init = torch.full((n_pixels,), 0.5, device=device)

        return t_init, beta_init, v_init

    def _emissivity_from_beta(self, beta: Tensor) -> Tensor:
        """Compute emissivity from B-spline coefficients.

        Args:
            beta: [..., K] B-spline coefficients.

        Returns:
            emissivity: [..., C] spectral emissivity.
        """
        return beta @ self.phi.T  # [..., K] @ [K, C] -> [..., C]

    def _compute_objective(
        self,
        s_obs: Tensor,
        temperature: Tensor,
        beta: Tensor,
        view_factor: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute full SLOT objective.

        Returns:
            (total_loss, s_model) for logging and gradient computation.
        """
        emissivity = self._emissivity_from_beta(beta)
        s_model = self.forward_model(temperature, emissivity, view_factor, s_sky, s_ground)

        # Data fidelity: ||S_obs - S_model||^2
        residual = ((s_obs - s_model) ** 2).sum(dim=-1).mean()

        # Smoothness: lambda/2 * ||D * beta||^2
        d_beta_prod = beta @ self.d_beta.T  # [N, K-2]
        smoothness = self.reg_lambda / 2.0 * (d_beta_prod**2).sum(dim=-1).mean()

        # Emissivity bounds: soft barrier for 0 < e < 1
        bound_violation = (
            torch.relu(-emissivity + 0.01).sum(dim=-1).mean()
            + torch.relu(emissivity - 0.99).sum(dim=-1).mean()
        )
        bound_penalty = self.constraint_mu * bound_violation

        total = residual + smoothness + bound_penalty
        return total, s_model

    @torch.no_grad()
    def _project_constraints(
        self,
        temperature: Tensor,
        beta: Tensor,
        view_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project variables onto feasible set.

        - T in [200, 500] K
        - V in [0, 1]
        - emissivity = Phi*beta in (0, 1) -> clamp beta accordingly
        """
        temperature = temperature.clamp(200.0, 500.0)
        view_factor = view_factor.clamp(0.0, 1.0)

        # Project emissivity into bounds by clamping and back-solving beta
        emissivity = self._emissivity_from_beta(beta)
        emissivity_clamped = emissivity.clamp(0.01, 0.99)
        if not torch.allclose(emissivity, emissivity_clamped, atol=1e-6):
            phi_pinv = torch.linalg.pinv(self.phi)
            beta = emissivity_clamped @ phi_pinv.T

        return temperature, beta, view_factor

    def decompose(
        self,
        s_obs: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
        verbose: bool = False,
    ) -> TexResult:
        """Run SLOT decomposition on observed spectral radiance.

        Args:
            s_obs: [N, C] or [H, W, C] observed hyperspectral thermal radiance.
            s_sky: [C] sky reference spectrum.
            s_ground: [C] ground reference spectrum.
            verbose: print convergence info.

        Returns:
            TexResult with all decomposed components.
        """
        # Handle spatial dimensions
        spatial_shape = None
        if s_obs.dim() == 3:
            spatial_shape = s_obs.shape[:2]
            s_obs = s_obs.reshape(-1, s_obs.shape[-1])

        # Initialize
        t_param, beta_param, v_param = self._initialize(s_obs, s_sky, s_ground)
        t_param = t_param.clone().requires_grad_(True)
        beta_param = beta_param.clone().requires_grad_(True)
        v_param = v_param.clone().requires_grad_(True)

        # Optimizer: L-BFGS
        optimizer = torch.optim.LBFGS(
            [t_param, beta_param, v_param],
            lr=self.lr,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        prev_loss = float("inf")
        final_iter = 0

        for iteration in range(self.max_iter):
            def closure():
                optimizer.zero_grad()
                loss, _ = self._compute_objective(
                    s_obs, t_param, beta_param, v_param, s_sky, s_ground
                )
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_val = loss.item()

            # Project onto feasible set
            with torch.no_grad():
                t_proj, b_proj, v_proj = self._project_constraints(
                    t_param, beta_param, v_param
                )
                t_param.data.copy_(t_proj)
                beta_param.data.copy_(b_proj)
                v_param.data.copy_(v_proj)

            if verbose and iteration % 10 == 0:
                print(f"  SLOT iter {iteration:4d}: objective={loss_val:.6f}")

            # Convergence check
            if abs(prev_loss - loss_val) < self.tolerance:
                final_iter = iteration + 1
                break
            prev_loss = loss_val
            final_iter = iteration + 1

        # Final results
        with torch.no_grad():
            emissivity = self._emissivity_from_beta(beta_param)
            _, s_recon = self._compute_objective(
                s_obs, t_param, beta_param, v_param, s_sky, s_ground
            )

            # Texture: broadband emissivity mean as grayscale
            texture = emissivity.mean(dim=-1)

            result = TexResult(
                temperature=t_param.detach(),
                emissivity=emissivity.detach(),
                view_factor=v_param.detach(),
                beta=beta_param.detach(),
                texture=texture,
                s_reconstructed=s_recon.detach(),
                objective=prev_loss,
                n_iterations=final_iter,
            )

        # Reshape back to spatial dims if needed
        if spatial_shape is not None:
            h, w = spatial_shape
            result.temperature = result.temperature.reshape(h, w)
            result.emissivity = result.emissivity.reshape(h, w, -1)
            result.view_factor = result.view_factor.reshape(h, w)
            result.beta = result.beta.reshape(h, w, -1)
            result.texture = result.texture.reshape(h, w)
            result.s_reconstructed = result.s_reconstructed.reshape(h, w, -1)

        return result

    def forward(
        self,
        s_obs: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> TexResult:
        """Forward pass (alias for decompose)."""
        return self.decompose(s_obs, s_sky, s_ground)


class HADARDecomposer(nn.Module):
    """Baseline HADAR method using material library lookup.

    HADAR pre-calibrates a library of (material_name, emissivity_spectrum) pairs
    and assigns each pixel to the closest library entry. This fails when intra-class
    material variation exceeds inter-class contrast (the ghosting effect).
    """

    def __init__(
        self,
        material_library: dict[str, Tensor] | None = None,
        wavenumber_grid: Tensor | None = None,
    ):
        """Initialize HADAR with material library.

        Args:
            material_library: dict mapping material name -> emissivity [C].
                If None, uses a default library of common materials.
            wavenumber_grid: [C] wavenumber grid.
        """
        super().__init__()
        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.register_buffer("wavenumber_grid", wavenumber_grid)

        if material_library is None:
            material_library = self._default_library()

        # Stack library into matrix [M, C]
        self.material_names = list(material_library.keys())
        lib_tensor = torch.stack(
            [material_library[name] for name in self.material_names]
        )
        self.register_buffer("library", lib_tensor)  # [M, C]

        self.forward_model = ThermalForwardModel(wavenumber_grid)

    def _default_library(self) -> dict[str, Tensor]:
        """Create a simple default material library with parametric emissivities."""
        c = len(default_wavenumber_grid())
        wg = default_wavenumber_grid()
        # Normalized wavenumber for creating spectral shapes
        v_norm = (wg - wg.min()) / (wg.max() - wg.min())

        return {
            "skin": torch.full((c,), 0.98),
            "fabric_cotton": 0.90 + 0.05 * torch.sin(2 * 3.14159 * v_norm),
            "metal_aluminum": 0.15 + 0.10 * v_norm,
            "vegetation": 0.95 - 0.03 * v_norm,
            "concrete": 0.92 + 0.02 * torch.sin(4 * 3.14159 * v_norm),
            "glass": 0.85 + 0.10 * v_norm,
        }

    def decompose(
        self,
        s_obs: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> TexResult:
        """HADAR decomposition via library lookup.

        For each pixel, find the library material that minimizes the
        rendering residual over (T, V).

        Args:
            s_obs: [N, C] observed radiance.
            s_sky: [C] sky reference.
            s_ground: [C] ground reference.

        Returns:
            TexResult with decomposed components.
        """
        n_pixels = s_obs.shape[0]
        n_materials = self.library.shape[0]
        device = s_obs.device

        best_loss = torch.full((n_pixels,), float("inf"), device=device)
        best_t = torch.full((n_pixels,), 300.0, device=device)
        best_e = torch.zeros(n_pixels, s_obs.shape[1], device=device)
        best_v = torch.full((n_pixels,), 0.5, device=device)

        # Grid search over materials and temperature
        t_candidates = torch.linspace(250.0, 350.0, 20, device=device)

        for m_idx in range(n_materials):
            e_m = self.library[m_idx].unsqueeze(0).expand(n_pixels, -1)  # [N, C]

            for t_val in t_candidates:
                t_tensor = torch.full((n_pixels,), t_val.item(), device=device)

                # Try V = 0.5 (simplified)
                v_tensor = torch.full((n_pixels,), 0.5, device=device)

                s_model = self.forward_model(t_tensor, e_m, v_tensor, s_sky, s_ground)
                loss = ((s_obs - s_model) ** 2).sum(dim=-1)  # [N]

                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                best_t = torch.where(improved, t_tensor, best_t)
                best_v = torch.where(improved, v_tensor, best_v)
                best_e = torch.where(improved.unsqueeze(-1), e_m, best_e)

        # Reconstruct
        s_recon = self.forward_model(best_t, best_e, best_v, s_sky, s_ground)
        texture = best_e.mean(dim=-1)

        return TexResult(
            temperature=best_t,
            emissivity=best_e,
            view_factor=best_v,
            beta=torch.zeros(n_pixels, 1, device=device),  # no B-spline
            texture=texture,
            s_reconstructed=s_recon,
            objective=best_loss.mean().item(),
            n_iterations=0,
        )

    def forward(
        self,
        s_obs: Tensor,
        s_sky: Tensor,
        s_ground: Tensor,
    ) -> TexResult:
        return self.decompose(s_obs, s_sky, s_ground)
