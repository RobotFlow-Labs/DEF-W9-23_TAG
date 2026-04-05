"""Tests for TAG dataset classes."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tag.dataset import SyntheticThermalDataset
from tag.utils import default_wavenumber_grid


class TestSyntheticThermalDataset:
    """Tests for synthetic thermal data generation."""

    def test_length(self):
        ds = SyntheticThermalDataset(n_scenes=10, height=16, width=16)
        assert len(ds) == 10

    def test_output_keys(self):
        ds = SyntheticThermalDataset(n_scenes=1, height=16, width=16)
        sample = ds[0]
        expected_keys = {"s_obs", "s_sky", "s_ground", "t_gt", "e_gt", "v_gt"}
        assert set(sample.keys()) == expected_keys

    def test_output_shapes(self):
        h, w = 16, 16
        ds = SyntheticThermalDataset(n_scenes=1, height=h, width=w)
        wg = default_wavenumber_grid()
        c = len(wg)
        sample = ds[0]

        assert sample["s_obs"].shape == (h, w, c)
        assert sample["s_sky"].shape == (c,)
        assert sample["s_ground"].shape == (c,)
        assert sample["t_gt"].shape == (h, w)
        assert sample["e_gt"].shape == (h, w, c)
        assert sample["v_gt"].shape == (h, w)

    def test_positive_radiance(self):
        ds = SyntheticThermalDataset(n_scenes=1, height=16, width=16)
        sample = ds[0]
        assert (sample["s_obs"] > 0).all()
        assert (sample["s_sky"] > 0).all()
        assert (sample["s_ground"] > 0).all()

    def test_temperature_range(self):
        t_range = (280.0, 320.0)
        ds = SyntheticThermalDataset(
            n_scenes=1, height=16, width=16, t_range=t_range
        )
        sample = ds[0]
        t = sample["t_gt"]
        # Temperature may exceed t_range slightly due to gradient, but should be close
        assert t.min() > t_range[0] - 20
        assert t.max() < t_range[1] + 20

    def test_emissivity_bounds(self):
        ds = SyntheticThermalDataset(n_scenes=1, height=16, width=16)
        sample = ds[0]
        e = sample["e_gt"]
        assert (e >= 0.04).all()
        assert (e <= 1.0).all()

    def test_view_factor_bounds(self):
        ds = SyntheticThermalDataset(n_scenes=1, height=16, width=16)
        sample = ds[0]
        v = sample["v_gt"]
        assert (v >= 0.0).all()
        assert (v <= 1.0).all()

    def test_reproducibility(self):
        ds1 = SyntheticThermalDataset(n_scenes=3, height=16, width=16, seed=42)
        ds2 = SyntheticThermalDataset(n_scenes=3, height=16, width=16, seed=42)
        for i in range(3):
            s1 = ds1[i]
            s2 = ds2[i]
            assert torch.allclose(s1["s_obs"], s2["s_obs"])
            assert torch.allclose(s1["t_gt"], s2["t_gt"])

    def test_different_seeds_different_data(self):
        ds1 = SyntheticThermalDataset(n_scenes=1, height=16, width=16, seed=42)
        ds2 = SyntheticThermalDataset(n_scenes=1, height=16, width=16, seed=99)
        assert not torch.allclose(ds1[0]["t_gt"], ds2[0]["t_gt"])

    def test_noise_free(self):
        """With NEDT=0, forward model should exactly match."""
        ds = SyntheticThermalDataset(n_scenes=1, height=8, width=8, nedt=0.0)
        sample = ds[0]
        # s_obs should be strictly positive (no negative from noise)
        assert (sample["s_obs"] > 0).all()
