"""Real thermal dataset adapters for TAG.

Converts single-band thermal imagery into pseudo-hyperspectral data
suitable for SLOT decomposition. Supports:
- VIVID++ thermal (16-bit raw, 320x256) — 71K+ images
- DroneVehicle-night (8-bit IR, various sizes)
- nuScenes night (camera data, subset selection)

For TAG, single-band thermal images are expanded to hyperspectral
by assuming parametric emissivity models with material variation.
This allows testing SLOT's ability to decompose TeX even when the
input is synthesized from single-band data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tag.utils import default_wavenumber_grid, planck_radiance


class VIVIDThermalDataset(Dataset):
    """VIVID++ thermal dataset adapter.

    VIVID++ contains 16-bit grayscale thermal images at 320x256 pixels.
    Raw values are in the range ~6000-8000 (likely decikelvin or ADC counts).
    We map these to physical temperatures and synthesize hyperspectral cubes.

    Path: /mnt/train-data/datasets/vivid_plus_plus/dataset/train/*/Thermal/*.png
    """

    def __init__(
        self,
        root_dir: str = "/mnt/train-data/datasets/vivid_plus_plus/dataset/train",
        split: str = "train",
        max_scenes: int | None = None,
        resize: tuple[int, int] | None = None,
        n_materials: int = 5,
        t_range: tuple[float, float] = (270.0, 340.0),
        wavenumber_grid: Tensor | None = None,
        seed: int = 42,
    ):
        """Initialize VIVID++ thermal dataset.

        Args:
            root_dir: path to VIVID++ train directory.
            split: 'train', 'val', or 'test' (splits the sequence list).
            max_scenes: maximum number of images to use.
            resize: optional (H, W) to resize images.
            n_materials: number of material emissivity templates.
            t_range: temperature mapping range in Kelvin.
            wavenumber_grid: [C] spectral grid.
            seed: random seed for material assignment.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.resize = resize
        self.n_materials = n_materials
        self.t_range = t_range
        self.seed = seed

        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.wavenumber_grid = wavenumber_grid
        self.n_channels = len(wavenumber_grid)

        # Find all thermal images
        self.image_paths = sorted(
            self.root_dir.rglob("Thermal/*.png")
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No thermal images found in {root_dir}")

        # Split: 90% train, 5% val, 5% test
        n_total = len(self.image_paths)
        n_train = int(0.9 * n_total)
        n_val = int(0.05 * n_total)

        if split == "train":
            self.image_paths = self.image_paths[:n_train]
        elif split == "val":
            self.image_paths = self.image_paths[n_train:n_train + n_val]
        elif split == "test":
            self.image_paths = self.image_paths[n_train + n_val:]

        if max_scenes is not None:
            self.image_paths = self.image_paths[:max_scenes]

        # Build material emissivity templates
        self.rng = np.random.RandomState(seed)
        self.material_spectra = self._build_material_spectra()

    def _build_material_spectra(self) -> Tensor:
        """Create parametric emissivity spectra for material templates."""
        v = self.wavenumber_grid.numpy()
        v_norm = (v - v.min()) / (v.max() - v.min())

        spectra = []
        # Common thermal materials
        bases = [0.98, 0.90, 0.85, 0.70, 0.50, 0.95, 0.88, 0.75]
        for i in range(self.n_materials):
            base = bases[i % len(bases)]
            freq = self.rng.uniform(0.5, 3.0)
            amp = self.rng.uniform(0.01, 0.06)
            phase = self.rng.uniform(0, 2 * math.pi)
            e = np.clip(
                base + amp * np.sin(2 * math.pi * freq * v_norm + phase),
                0.05, 0.99,
            ).astype(np.float32)
            spectra.append(torch.from_numpy(e))

        return torch.stack(spectra)  # [n_materials, C]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(str(path))

        if self.resize is not None:
            img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        # Map raw 16-bit values to temperature
        # VIVID++ raw: ~6000-8000 range → map to Kelvin
        raw_min, raw_max = 5000.0, 9000.0
        t_norm = (arr - raw_min) / (raw_max - raw_min)
        t_norm = np.clip(t_norm, 0, 1)
        t_map = torch.from_numpy(
            t_norm * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        )

        # Assign emissivity based on temperature clustering
        # (simple k-means-like assignment to material templates)
        n_levels = self.n_materials
        t_flat = t_map.flatten()
        t_quantiles = torch.quantile(t_flat, torch.linspace(0, 1, n_levels + 1))

        e_map = torch.zeros(h, w, self.n_channels)
        for mi in range(n_levels):
            mask = (t_map >= t_quantiles[mi]) & (t_map <= t_quantiles[mi + 1])
            if mask.any():
                e_map[mask] = self.material_spectra[mi]

        # Add pixel-level emissivity variation
        rng_local = np.random.RandomState(self.seed + idx)
        noise_e = torch.from_numpy(
            rng_local.normal(0, 0.015, (h, w, self.n_channels)).astype(np.float32)
        )
        e_map = (e_map + noise_e).clamp(0.05, 0.99)

        # View factor from spatial gradient
        v_map = torch.from_numpy(
            np.clip(0.5 + rng_local.normal(0, 0.08, (h, w)).astype(np.float32), 0, 1)
        )

        # Reference spectra
        wg = self.wavenumber_grid
        s_sky = planck_radiance(wg, torch.tensor(235.0)).squeeze()
        s_ground = planck_radiance(wg, torch.tensor(295.0)).squeeze()

        # Forward model → hyperspectral cube
        b_planck = planck_radiance(wg, t_map)
        x_ambient = v_map.unsqueeze(-1) * s_sky + (1 - v_map.unsqueeze(-1)) * s_ground
        s_obs = e_map * b_planck + (1 - e_map) * x_ambient

        # Add realistic sensor noise
        t_mean = t_map.mean()
        db_dt = planck_radiance(wg, t_mean + 0.5) - planck_radiance(wg, t_mean - 0.5)
        noise = torch.from_numpy(
            rng_local.randn(h, w, self.n_channels).astype(np.float32)
        ) * 0.03 * db_dt.abs()
        s_obs = (s_obs + noise).clamp(min=1e-20)

        return {
            "s_obs": s_obs,
            "s_sky": s_sky,
            "s_ground": s_ground,
            "t_gt": t_map,
            "e_gt": e_map,
            "v_gt": v_map,
            "path": str(path),
        }


class DroneVehicleNightDataset(Dataset):
    """DroneVehicle-night thermal dataset adapter.

    IR images from drone-mounted thermal cameras, nighttime vehicle scenes.
    Path: /mnt/forge-data/datasets/wave9/drones/DroneVehicle-night/
    """

    def __init__(
        self,
        root_dir: str = "/mnt/forge-data/datasets/wave9/drones/DroneVehicle-night",
        split: str = "train",
        max_scenes: int | None = None,
        target_size: tuple[int, int] = (256, 320),
        wavenumber_grid: Tensor | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.seed = seed

        if wavenumber_grid is None:
            wavenumber_grid = default_wavenumber_grid()
        self.wavenumber_grid = wavenumber_grid
        self.n_channels = len(wavenumber_grid)

        # Find IR images (typical structure: infrared/*.jpg or *.png)
        self.image_paths = sorted(
            list(self.root_dir.rglob("*.jpg"))
            + list(self.root_dir.rglob("*.png"))
            + list(self.root_dir.rglob("*.bmp"))
        )

        # Split
        n_total = len(self.image_paths)
        n_train = int(0.9 * n_total)
        n_val = int(0.05 * n_total)

        if split == "train":
            self.image_paths = self.image_paths[:n_train]
        elif split == "val":
            self.image_paths = self.image_paths[n_train:n_train + n_val]
        elif split == "test":
            self.image_paths = self.image_paths[n_train + n_val:]

        if max_scenes is not None:
            self.image_paths = self.image_paths[:max_scenes]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(str(path)).convert("L")
        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w = arr.shape

        # Map to temperature (nighttime: cooler range)
        t_map = torch.from_numpy(arr * 50.0 + 265.0)  # 265-315K

        # Simple emissivity model
        wg = self.wavenumber_grid
        c = len(wg)
        e_map = torch.full((h, w, c), 0.92)

        # Add variation based on intensity
        rng_local = np.random.RandomState(self.seed + idx)
        noise_e = torch.from_numpy(
            rng_local.normal(0, 0.02, (h, w, c)).astype(np.float32)
        )
        e_map = (e_map + noise_e).clamp(0.05, 0.99)

        v_map = torch.from_numpy(
            np.clip(0.6 + rng_local.normal(0, 0.05, (h, w)).astype(np.float32), 0, 1)
        )

        s_sky = planck_radiance(wg, torch.tensor(230.0)).squeeze()
        s_ground = planck_radiance(wg, torch.tensor(285.0)).squeeze()

        b_planck = planck_radiance(wg, t_map)
        x_ambient = v_map.unsqueeze(-1) * s_sky + (1 - v_map.unsqueeze(-1)) * s_ground
        s_obs = e_map * b_planck + (1 - e_map) * x_ambient
        s_obs = s_obs.clamp(min=1e-20)

        return {
            "s_obs": s_obs,
            "s_sky": s_sky,
            "s_ground": s_ground,
            "t_gt": t_map,
            "e_gt": e_map,
            "v_gt": v_map,
            "path": str(path),
        }


class CombinedThermalDataset(Dataset):
    """Combines multiple thermal datasets for multi-source training.

    Interleaves samples from synthetic, VIVID++, DroneVehicle, etc.
    """

    def __init__(self, datasets: list[Dataset], weights: list[float] | None = None):
        """Initialize combined dataset.

        Args:
            datasets: list of dataset instances.
            weights: sampling weights per dataset (default: proportional to size).
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)

        # Build index mapping: global_idx -> (dataset_idx, local_idx)
        self.index_map = []
        for di, ds in enumerate(datasets):
            for li in range(len(ds)):
                self.index_map.append((di, li))

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        di, li = self.index_map[idx]
        sample = self.datasets[di][li]
        sample["dataset_idx"] = di
        return sample
