"""Tests for TAG export pipeline."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA GPU")


class TestForwardModule:
    """Tests for exportable forward module."""

    def test_output_shapes(self):
        from tag.export import TAGForwardModule
        from tag.utils import default_wavenumber_grid, planck_radiance

        wg = default_wavenumber_grid()
        model = TAGForwardModule(n_knots=10, wavenumber_grid=wg)
        n = 32
        k = model.phi.shape[1]

        t = torch.full((n,), 300.0)
        beta = torch.full((n, k), 0.04)
        v = torch.full((n,), 0.5)
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))

        s_model, emissivity, texture = model(t, beta, v, s_sky, s_ground)
        assert s_model.shape == (n, len(wg))
        assert emissivity.shape == (n, len(wg))
        assert texture.shape == (n,)

    def test_positive_radiance(self):
        from tag.export import TAGForwardModule
        from tag.utils import default_wavenumber_grid, planck_radiance

        wg = default_wavenumber_grid()
        model = TAGForwardModule(n_knots=10, wavenumber_grid=wg)
        k = model.phi.shape[1]

        t = torch.full((4,), 300.0)
        beta = torch.full((4, k), 0.04)
        v = torch.full((4,), 0.5)
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))

        s_model, _, _ = model(t, beta, v, s_sky, s_ground)
        assert (s_model > 0).all()


class TestDecomposeModule:
    """Tests for exportable decompose module."""

    def test_output_shapes(self):
        from tag.export import TAGDecomposeModule
        from tag.utils import default_wavenumber_grid, planck_radiance

        wg = default_wavenumber_grid()
        model = TAGDecomposeModule(n_knots=10, wavenumber_grid=wg)
        n = 16
        c = len(wg)
        k = model.phi.shape[1]

        s_obs = torch.randn(n, c).abs() * 1e-5
        t = torch.full((n,), 300.0)
        beta = torch.full((n, k), 0.04)
        v = torch.full((n,), 0.5)
        s_sky = planck_radiance(wg, torch.tensor(240.0))
        s_ground = planck_radiance(wg, torch.tensor(290.0))

        obj, s_model, emissivity = model(s_obs, t, beta, v, s_sky, s_ground)
        assert obj.dim() == 0  # scalar
        assert s_model.shape == (n, c)
        assert emissivity.shape == (n, c)


class TestONNXExport:
    """Tests for ONNX export."""

    @skip_no_cuda
    def test_forward_onnx(self, tmp_path):
        from tag.export import TAGForwardModule, export_onnx
        from tag.utils import default_wavenumber_grid

        device = torch.device("cuda:0")
        wg = default_wavenumber_grid(device)
        model = TAGForwardModule(n_knots=10, wavenumber_grid=wg).to(device)

        onnx_path = tmp_path / "test_forward.onnx"
        result = export_onnx(model, onnx_path, n_pixels=64, device=device)
        assert result.exists()
        assert result.stat().st_size > 0
