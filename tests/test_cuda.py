"""Tests for CUDA-accelerated TAG components."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA GPU")


class TestBSplineCUDA:
    """Tests for vectorized B-spline basis."""

    def test_shape_matches_reference(self):
        from tag.cuda.kernels import bspline_basis_cuda
        from tag.utils import cubic_bspline_basis, default_wavenumber_grid

        wg = default_wavenumber_grid()
        n_knots = 20

        ref = cubic_bspline_basis(wg, n_knots)
        cuda_result = bspline_basis_cuda(wg, n_knots)

        assert ref.shape == cuda_result.shape

    def test_values_match_reference(self):
        from tag.cuda.kernels import bspline_basis_cuda
        from tag.utils import cubic_bspline_basis, default_wavenumber_grid

        wg = default_wavenumber_grid()
        n_knots = 20

        ref = cubic_bspline_basis(wg, n_knots)
        cuda_result = bspline_basis_cuda(wg, n_knots)

        assert torch.allclose(ref, cuda_result, atol=1e-5), \
            f"Max diff: {(ref - cuda_result).abs().max():.6e}"

    def test_partition_of_unity(self):
        from tag.cuda.kernels import bspline_basis_cuda
        from tag.utils import default_wavenumber_grid

        wg = default_wavenumber_grid()
        phi = bspline_basis_cuda(wg, 20)
        row_sums = phi.sum(dim=-1)
        interior = row_sums[2:-2]
        assert torch.allclose(interior, torch.ones_like(interior), atol=0.1)

    @skip_no_cuda
    def test_cuda_device(self):
        from tag.cuda.kernels import bspline_basis_cuda
        from tag.utils import default_wavenumber_grid

        wg = default_wavenumber_grid(device=torch.device("cuda:0"))
        phi = bspline_basis_cuda(wg, 20)
        assert phi.device.type == "cuda"


class TestPlanckCUDA:
    """Tests for compiled Planck radiance."""

    @skip_no_cuda
    def test_matches_reference(self):
        from tag.cuda.kernels import planck_radiance_cuda
        from tag.utils import default_wavenumber_grid, planck_radiance

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)
        t = torch.tensor([280.0, 300.0, 320.0], device=device)

        ref = planck_radiance(wg, t)
        cuda_result = planck_radiance_cuda(wg, t)

        assert torch.allclose(ref, cuda_result, rtol=1e-4)

    @skip_no_cuda
    def test_batched_shape(self):
        from tag.cuda.kernels import planck_radiance_cuda
        from tag.utils import default_wavenumber_grid

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)
        t = torch.rand(64, 64, device=device) * 60 + 270

        result = planck_radiance_cuda(wg, t)
        assert result.shape == (64, 64, len(wg))


class TestVoronoiCUDA:
    """Tests for vectorized Voronoi assignment."""

    @skip_no_cuda
    def test_shape(self):
        from tag.cuda.kernels import voronoi_assign_cuda

        device = torch.device("cuda:0")
        centers = torch.rand(5, 2, device=device)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)

        result = voronoi_assign_cuda(64, 64, centers, values)
        assert result.shape == (64, 64)

    @skip_no_cuda
    def test_multidim_values(self):
        from tag.cuda.kernels import voronoi_assign_cuda

        device = torch.device("cuda:0")
        centers = torch.rand(3, 2, device=device)
        values = torch.rand(3, 67, device=device)

        result = voronoi_assign_cuda(32, 32, centers, values)
        assert result.shape == (32, 32, 67)


class TestFusedForward:
    """Tests for fused SLOT forward model."""

    @skip_no_cuda
    def test_output_shape(self):
        from tag.cuda.kernels import bspline_basis_cuda, fused_slot_forward_cuda
        from tag.utils import default_wavenumber_grid, planck_radiance

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)
        phi = bspline_basis_cuda(wg, 20)
        n = 100
        k = phi.shape[1]

        t = torch.full((n,), 300.0, device=device)
        beta = torch.full((n, k), 0.04, device=device)
        v = torch.full((n,), 0.5, device=device)
        s_sky = planck_radiance(wg, torch.tensor(240.0, device=device))
        s_ground = planck_radiance(wg, torch.tensor(290.0, device=device))

        s_model, emissivity = fused_slot_forward_cuda(t, beta, v, phi, wg, s_sky, s_ground)
        assert s_model.shape == (n, len(wg))
        assert emissivity.shape == (n, len(wg))


class TestSLOTDecomposerCUDA:
    """Tests for full CUDA SLOT decomposer."""

    @skip_no_cuda
    def test_initialization(self):
        from tag.cuda.slot_cuda import SLOTDecomposerCUDA

        device = torch.device("cuda:0")
        decomposer = SLOTDecomposerCUDA(n_knots=10, max_iter=5).to(device)
        assert decomposer.n_basis > 0

    @skip_no_cuda
    def test_noise_free_recovery(self):
        from tag.cuda.slot_cuda import SLOTDecomposerCUDA
        from tag.model import ThermalForwardModel
        from tag.utils import default_wavenumber_grid, planck_radiance

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)
        n = 16

        t_gt = torch.full((n,), 300.0, device=device)
        e_gt = torch.full((n, len(wg)), 0.9, device=device)
        v_gt = torch.full((n,), 0.5, device=device)
        s_sky = planck_radiance(wg, torch.tensor(240.0, device=device))
        s_ground = planck_radiance(wg, torch.tensor(290.0, device=device))

        fm = ThermalForwardModel(wg).to(device)
        s_obs = fm(t_gt, e_gt, v_gt, s_sky, s_ground)

        decomposer = SLOTDecomposerCUDA(
            n_knots=10, max_iter=50, reg_lambda=0.1
        ).to(device)
        result = decomposer.decompose(s_obs, s_sky, s_ground)

        t_error = (result.temperature - t_gt).abs().mean()
        assert t_error < 10.0, f"Temperature error too high: {t_error:.2f}K"

    @skip_no_cuda
    def test_spatial_input(self):
        """Test with [H, W, C] input."""
        from tag.cuda.slot_cuda import SLOTDecomposerCUDA
        from tag.cuda.kernels import generate_synthetic_scene_cuda
        from tag.utils import default_wavenumber_grid

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)

        scene = generate_synthetic_scene_cuda(
            8, 8, 3, wg, (280.0, 320.0), 0.03, device=device,
        )

        decomposer = SLOTDecomposerCUDA(
            n_knots=10, max_iter=20, chunk_size=32,
        ).to(device)
        result = decomposer.decompose(scene["s_obs"], scene["s_sky"], scene["s_ground"])

        assert result.temperature.shape == (8, 8)
        assert result.emissivity.shape == (8, 8, len(wg))
