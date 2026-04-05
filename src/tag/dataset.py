"""Dataset classes for TAG: synthetic thermal data generation and IR image adapters.

TAG requires hyperspectral thermal data (67 spectral bands, 870-1269 cm^-1).
Since no public dataset exists, we generate synthetic scenes from known
temperature maps, parametric emissivity spectra, and view factors.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tag.utils import (
    default_wavenumber_grid,
    planck_radiance,
)


class SyntheticThermalDataset(Dataset):
    """Generate synthetic hyperspectral thermal scenes.

    Each sample contains:
      - s_obs: [H, W, C] observed spectral radiance (with noise)
      - s_sky: [C] sky reference spectrum
      - s_ground: [C] ground reference spectrum
      - t_gt: [H, W] ground truth temperature
      - e_gt: [H, W, C] ground truth emissivity
      - v_gt: [H, W] ground truth view factor
    """

    def __init__(
        self,
        n_scenes: int = 100,
        height: int = 64,
        width: int = 64,
        n_materials: int = 5,
        t_range: tuple[float, float] = (270.0, 330.0),
        nedt: float = 0.03,
        seed: int = 42,
        wavenumber_grid: Tensor | None = None,
    ):
        """Initialize synthetic dataset.

        Args:
            n_scenes: number of scenes to generate.
            height: scene height in pixels.
            width: scene width in pixels.
            n_materials: number of distinct material regions per scene.
            t_range: temperature range in Kelvin.
            nedt: noise equivalent differential temperature in K.
            seed: random seed.
            wavenumber_grid: [C] wavenumber grid.
        """
        super().__init__()
        self.n_scenes = n_scenes
        self.height = height
        self.width = width
        self.n_materials = n_materials
        self.t_range = t_range
        self.nedt = nedt
        self.seed = seed

        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.wavenumber_grid = wavenumber_grid
        self.n_channels = len(wavenumber_grid)

        # Pre-generate all scenes for reproducibility
        self.rng = np.random.RandomState(seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(seed)
        self.scenes = self._generate_all()

    def _random_emissivity_spectrum(self, n_spectra: int) -> Tensor:
        """Generate random physically-plausible emissivity spectra.

        Uses a mixture of graybody + spectral variation model.

        Args:
            n_spectra: number of spectra to generate.

        Returns:
            [n_spectra, C] emissivity spectra in (0.05, 0.99).
        """
        v = self.wavenumber_grid.numpy()
        v_norm = (v - v.min()) / (v.max() - v.min())

        spectra = []
        for _ in range(n_spectra):
            # Base emissivity
            base = self.rng.uniform(0.3, 0.98)
            # Spectral variation: low-frequency sinusoidal
            n_modes = self.rng.randint(1, 4)
            variation = np.zeros_like(v_norm)
            for _ in range(n_modes):
                freq = self.rng.uniform(0.5, 3.0)
                amp = self.rng.uniform(0.01, 0.08)
                phase = self.rng.uniform(0, 2 * math.pi)
                variation += amp * np.sin(2 * math.pi * freq * v_norm + phase)

            e = np.clip(base + variation, 0.05, 0.99).astype(np.float32)
            spectra.append(torch.from_numpy(e))

        return torch.stack(spectra)  # [n_spectra, C]

    def _generate_scene(self) -> dict[str, Tensor]:
        """Generate a single synthetic thermal scene."""
        h, w, c = self.height, self.width, self.n_channels
        wg = self.wavenumber_grid

        # Temperature map: piecewise smooth regions
        t_map = np.zeros((h, w), dtype=np.float32)
        n_regions = self.rng.randint(3, self.n_materials + 1)

        # Create Voronoi-like regions
        centers = self.rng.rand(n_regions, 2)
        temps = self.rng.uniform(self.t_range[0], self.t_range[1], n_regions)

        for y in range(h):
            for x in range(w):
                dists = np.sqrt(
                    (centers[:, 0] - y / h) ** 2 + (centers[:, 1] - x / w) ** 2
                )
                t_map[y, x] = temps[np.argmin(dists)]

        # Add smooth temperature gradient
        grad_y = np.linspace(0, self.rng.uniform(-5, 5), h).reshape(-1, 1)
        t_map += grad_y
        t_gt = torch.from_numpy(t_map)

        # Emissivity: assign spectra to regions
        material_spectra = self._random_emissivity_spectrum(n_regions)
        e_map = np.zeros((h, w, c), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                dists = np.sqrt(
                    (centers[:, 0] - y / h) ** 2 + (centers[:, 1] - x / w) ** 2
                )
                region_idx = np.argmin(dists)
                e_map[y, x] = material_spectra[region_idx].numpy()

        # Add intra-material variation (the ghosting source)
        noise_e = self.rng.normal(0, 0.02, (h, w, c)).astype(np.float32)
        e_map = np.clip(e_map + noise_e, 0.05, 0.99)
        e_gt = torch.from_numpy(e_map)

        # View factor: smooth spatial field
        v_base = self.rng.uniform(0.2, 0.8)
        v_var = self.rng.normal(0, 0.1, (h, w)).astype(np.float32)
        v_map = np.clip(v_base + v_var, 0.0, 1.0)
        v_gt = torch.from_numpy(v_map)

        # Sky and ground reference spectra
        t_sky = self.rng.uniform(220.0, 260.0)
        t_ground = self.rng.uniform(280.0, 310.0)
        s_sky = planck_radiance(wg, torch.tensor(t_sky)).squeeze()
        s_ground = planck_radiance(wg, torch.tensor(t_ground)).squeeze()

        # Forward model: S = e * B(T) + (1-e) * (V*S_sky + (1-V)*S_ground)
        b_planck = planck_radiance(wg, t_gt)  # [H, W, C]
        x_ambient = v_gt.unsqueeze(-1) * s_sky + (1 - v_gt.unsqueeze(-1)) * s_ground
        s_clean = e_gt * b_planck + (1 - e_gt) * x_ambient

        # Add sensor noise (NEDT)
        if self.nedt > 0:
            # Convert NEDT to radiance noise using dB/dT at mean T
            t_mean = t_gt.mean()
            db_dt = planck_radiance(wg, t_mean + 0.5) - planck_radiance(wg, t_mean - 0.5)
            noise_std = self.nedt * db_dt.abs()
            noise = torch.randn(s_clean.shape, generator=self.torch_gen) * noise_std
            s_obs = s_clean + noise
        else:
            s_obs = s_clean.clone()

        # Ensure positive radiance
        s_obs = s_obs.clamp(min=1e-20)

        return {
            "s_obs": s_obs,
            "s_sky": s_sky,
            "s_ground": s_ground,
            "t_gt": t_gt,
            "e_gt": e_gt,
            "v_gt": v_gt,
        }

    def _generate_all(self) -> list[dict[str, Tensor]]:
        """Pre-generate all scenes."""
        return [self._generate_scene() for _ in range(self.n_scenes)]

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.scenes[idx]


class IRImageDataset(Dataset):
    """Adapter for single-band IR images (e.g. NUAA-SIRST).

    Converts single-band thermal images into pseudo-hyperspectral data
    by assuming a parametric emissivity model. Useful for testing the
    pipeline on existing IR datasets.
    """

    def __init__(
        self,
        image_dir: str | Path,
        assumed_emissivity: float = 0.95,
        t_range: tuple[float, float] = (270.0, 330.0),
        wavenumber_grid: Tensor | None = None,
    ):
        """Initialize IR image adapter.

        Args:
            image_dir: directory containing IR images (.png, .jpg, .npy).
            assumed_emissivity: assumed constant emissivity for all pixels.
            t_range: temperature range for mapping pixel intensities.
            wavenumber_grid: [C] wavenumber grid.
        """
        super().__init__()
        self.image_dir = Path(image_dir)
        self.assumed_emissivity = assumed_emissivity
        self.t_range = t_range

        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.wavenumber_grid = wavenumber_grid

        # Find all image files
        self.image_paths = sorted(
            list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.npy"))
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path = self.image_paths[idx]

        # Load image
        if path.suffix == ".npy":
            img = torch.from_numpy(np.load(str(path))).float()
        else:
            from PIL import Image

            img = Image.open(str(path)).convert("L")
            img = torch.from_numpy(np.array(img)).float()

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        h, w = img.shape

        # Map pixel intensity to temperature
        t_map = img * (self.t_range[1] - self.t_range[0]) + self.t_range[0]

        # Create constant emissivity
        c = len(self.wavenumber_grid)
        e_map = torch.full((h, w, c), self.assumed_emissivity)

        # View factor: uniform
        v_map = torch.full((h, w), 0.5)

        # Sky/ground references (assume standard conditions)
        s_sky = planck_radiance(self.wavenumber_grid, torch.tensor(240.0)).squeeze()
        s_ground = planck_radiance(self.wavenumber_grid, torch.tensor(290.0)).squeeze()

        # Forward model to generate pseudo-hyperspectral
        b_planck = planck_radiance(self.wavenumber_grid, t_map)
        x_ambient = v_map.unsqueeze(-1) * s_sky + (1 - v_map.unsqueeze(-1)) * s_ground
        s_obs = e_map * b_planck + (1 - e_map) * x_ambient

        return {
            "s_obs": s_obs,
            "s_sky": s_sky,
            "s_ground": s_ground,
            "t_gt": t_map,
            "e_gt": e_map,
            "v_gt": v_map,
        }
