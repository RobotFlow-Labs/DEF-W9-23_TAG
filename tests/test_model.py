"""Tests for TAG model components: Planck, B-spline, SLOT, forward model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tag.losses import (
    average_gradient,
    emissivity_bound_penalty,
    information_entropy,
    rendering_residual,
    smoothness_penalty,
    spatial_frequency,
    standard_deviation,
)
from tag.model import SLOTDecomposer, ThermalForwardModel
from tag.utils import (
    cubic_bspline_basis,
    default_wavenumber_grid,
    planck_radiance,
    second_order_diff_operator,
)


class TestPlanckRadiance:
    """Tests for Planck's law implementation."""

    def test_output_shape(self):
        wg = default_wavenumber_grid()
        t = torch.tensor([300.0, 310.0, 320.0])
        result = planck_radiance(wg, t)
        assert result.shape == (3, len(wg))

    def test_positive_radiance(self):
        wg = default_wavenumber_grid()
        t = torch.tensor([250.0, 300.0, 350.0])
        result = planck_radiance(wg, t)
        assert (result > 0).all()

    def test_higher_temperature_more_radiance(self):
        wg = default_wavenumber_grid()
        t_low = torch.tensor(280.0)
        t_high = torch.tensor(320.0)
        r_low = planck_radiance(wg, t_low)
        r_high = planck_radiance(wg, t_high)
        # Higher T should produce more radiance at all wavenumbers in LWIR
        assert (r_high > r_low).all()

    def test_scalar_temperature(self):
        wg = default_wavenumber_grid()
        t = torch.tensor(300.0)
        result = planck_radiance(wg, t)
        assert result.shape == (len(wg),)

    def test_reference_value(self):
        """Verify Planck at 300K, 1000 cm^-1 against known value.

        B_v(1000 cm^-1, 300K) ~ 9.92e-6 W/(m^2 sr cm^-1)
        C1 = 2*h*c_cm^2 ~ 1.191e-12, C2 = h*c_cm/kB ~ 1.4388 cm*K
        exponent = C2*v/T = 1.4388*1000/300 = 4.796
        B = C1*v^3/(exp(4.796)-1) = 1.191e-12 * 1e9 / 120.06 ~ 9.92e-6
        """
        wg = torch.tensor([1000.0])
        t = torch.tensor(300.0)
        result = planck_radiance(wg, t)
        assert 5e-6 < result.item() < 5e-5


class TestBSplineBasis:
    """Tests for cubic B-spline basis construction."""

    def test_output_shape(self):
        wg = default_wavenumber_grid()
        n_knots = 20
        phi = cubic_bspline_basis(wg, n_knots)
        # K = n_knots + order - 2 = 20 + 4 - 2 = 22
        expected_k = n_knots + 4 - 2
        assert phi.shape == (len(wg), expected_k)

    def test_partition_of_unity(self):
        """B-spline basis should sum to ~1 at interior points."""
        wg = default_wavenumber_grid()
        phi = cubic_bspline_basis(wg, 20)
        row_sums = phi.sum(dim=-1)
        # Interior points should be close to 1
        interior = row_sums[2:-2]
        assert torch.allclose(interior, torch.ones_like(interior), atol=0.1)

    def test_non_negative(self):
        wg = default_wavenumber_grid()
        phi = cubic_bspline_basis(wg, 20)
        assert (phi >= -1e-6).all()

    def test_different_n_knots(self):
        wg = default_wavenumber_grid()
        for n_knots in [5, 10, 20, 30]:
            phi = cubic_bspline_basis(wg, n_knots)
            expected_k = n_knots + 4 - 2
            assert phi.shape == (len(wg), expected_k)


class TestDiffOperator:
    """Tests for second-order difference operator."""

    def test_shape(self):
        d = second_order_diff_operator(22)
        assert d.shape == (20, 22)

    def test_constant_beta_zero_penalty(self):
        """Constant beta should have zero second-order differences."""
        d = second_order_diff_operator(22)
        beta = torch.ones(1, 22)
        result = beta @ d.T
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_linear_beta_zero_penalty(self):
        """Linear beta should have zero second-order differences."""
        d = second_order_diff_operator(22)
        beta = torch.linspace(0, 1, 22).unsqueeze(0)
        result = beta @ d.T
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_oscillatory_beta_nonzero(self):
        """Oscillatory beta should have nonzero penalty."""
        d = second_order_diff_operator(22)
        k = torch.arange(22).float()
        beta = torch.sin(k * 3.14).unsqueeze(0)
        result = beta @ d.T
        assert result.abs().sum() > 0.1


class TestForwardModel:
    """Tests for ThermalForwardModel."""

    def test_output_shape(self):
        wg = default_wavenumber_grid()
        model = ThermalForwardModel(wg)
        n = 10
        t = torch.full((n,), 300.0)
        e = torch.full((n, len(wg)), 0.9)
        v = torch.full((n,), 0.5)
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))
        result = model(t, e, v, s_sky, s_ground)
        assert result.shape == (n, len(wg))

    def test_positive_radiance(self):
        wg = default_wavenumber_grid()
        model = ThermalForwardModel(wg)
        t = torch.tensor([300.0])
        e = torch.full((1, len(wg)), 0.9)
        v = torch.tensor([0.5])
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))
        result = model(t, e, v, s_sky, s_ground)
        assert (result > 0).all()


class TestSLOTDecomposer:
    """Tests for SLOT decomposer."""

    def test_initialization(self):
        decomposer = SLOTDecomposer(n_knots=10, max_iter=5)
        assert decomposer.n_basis > 0
        assert decomposer.phi.shape[0] > 0

    def test_noise_free_recovery(self):
        """SLOT should recover T/e/V well on noise-free data."""
        wg = default_wavenumber_grid()
        n = 16  # small for speed

        # Known ground truth
        t_gt = torch.full((n,), 300.0)
        e_gt = torch.full((n, len(wg)), 0.9)
        v_gt = torch.full((n,), 0.5)
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))

        # Forward model
        fm = ThermalForwardModel(wg)
        s_obs = fm(t_gt, e_gt, v_gt, s_sky, s_ground)

        # SLOT decomposition
        decomposer = SLOTDecomposer(n_knots=10, max_iter=50, reg_lambda=0.1)
        result = decomposer.decompose(s_obs, s_sky, s_ground)

        # Temperature should be close
        t_error = (result.temperature - t_gt).abs().mean()
        assert t_error < 10.0, f"Temperature RMSE too high: {t_error:.2f}K"


class TestLossFunctions:
    """Tests for loss functions."""

    def test_zero_residual(self):
        s = torch.randn(10, 67)
        assert rendering_residual(s, s).item() < 1e-10

    def test_constant_beta_zero_smoothness(self):
        d = second_order_diff_operator(22)
        beta = torch.ones(10, 22)
        assert smoothness_penalty(beta, d).item() < 1e-10

    def test_valid_emissivity_no_bound_penalty(self):
        e = torch.full((10, 67), 0.5)
        assert emissivity_bound_penalty(e).item() < 1e-6

    def test_invalid_emissivity_has_penalty(self):
        e = torch.full((10, 67), 1.5)  # out of bounds
        assert emissivity_bound_penalty(e).item() > 0.1


class TestImageQualityMetrics:
    """Tests for EN, AG, SF, SD metrics."""

    def test_entropy_uniform(self):
        """Uniform image should have maximum entropy."""
        img = torch.rand(64, 64)
        en = information_entropy(img)
        assert en.item() > 0

    def test_entropy_constant(self):
        """Constant image should have near-zero entropy."""
        img = torch.full((64, 64), 0.5)
        en = information_entropy(img)
        assert en.item() < 1.0

    def test_gradient_constant(self):
        """Constant image should have zero gradient."""
        img = torch.full((64, 64), 0.5)
        ag = average_gradient(img)
        assert ag.item() < 1e-6

    def test_sf_constant(self):
        """Constant image should have zero spatial frequency."""
        img = torch.full((64, 64), 0.5)
        sf = spatial_frequency(img)
        assert sf.item() < 1e-6

    def test_sd_constant(self):
        """Constant image should have zero standard deviation."""
        img = torch.full((64, 64), 0.5)
        sd = standard_deviation(img)
        assert sd.item() < 1e-6

    def test_sd_nonzero(self):
        """Random image should have nonzero standard deviation."""
        img = torch.rand(64, 64)
        sd = standard_deviation(img)
        assert sd.item() > 0.1
