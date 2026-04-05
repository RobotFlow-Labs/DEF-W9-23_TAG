"""Evaluation pipeline for TAG: image quality metrics and reconstruction accuracy.

Reproduces Table 1 metrics (EN, AG, SF, SD) and measures TeX reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from tag.dataset import SyntheticThermalDataset
from tag.losses import (
    average_gradient,
    information_entropy,
    spatial_frequency,
    spectral_angle_mapper,
    standard_deviation,
)
from tag.model import HADARDecomposer, SLOTDecomposer
from tag.utils import default_wavenumber_grid, seed_everything

PROJECT = "project_tag"
ARTIFACTS = "/mnt/artifacts-datai"


def evaluate_decomposition(
    result_temperature: torch.Tensor,
    result_emissivity: torch.Tensor,
    result_view_factor: torch.Tensor,
    result_texture: torch.Tensor,
    gt_temperature: torch.Tensor,
    gt_emissivity: torch.Tensor,
    gt_view_factor: torch.Tensor,
) -> dict[str, float]:
    """Compute all evaluation metrics for a single scene.

    Args:
        result_*: decomposition outputs.
        gt_*: ground truth values.

    Returns:
        Dict of metric name -> value.
    """
    metrics = {}

    # Reconstruction accuracy
    metrics["t_rmse"] = torch.sqrt(
        ((result_temperature - gt_temperature) ** 2).mean()
    ).item()
    metrics["t_mae"] = (result_temperature - gt_temperature).abs().mean().item()

    metrics["e_mae"] = (result_emissivity - gt_emissivity).abs().mean().item()
    metrics["e_rmse"] = torch.sqrt(
        ((result_emissivity - gt_emissivity) ** 2).mean()
    ).item()

    metrics["v_rmse"] = torch.sqrt(
        ((result_view_factor - gt_view_factor) ** 2).mean()
    ).item()

    # Spectral angle mapper for emissivity
    if result_emissivity.dim() == 3:
        pred_flat = result_emissivity.reshape(-1, result_emissivity.shape[-1])
        gt_flat = gt_emissivity.reshape(-1, gt_emissivity.shape[-1])
    else:
        pred_flat = result_emissivity
        gt_flat = gt_emissivity
    metrics["sam_rad"] = spectral_angle_mapper(pred_flat, gt_flat).item()

    # Image quality metrics on texture map
    texture = result_texture.float()
    if texture.dim() == 2:
        metrics["en"] = information_entropy(texture).item()
        metrics["ag"] = average_gradient(texture).item()
        metrics["sf"] = spatial_frequency(texture).item()
        metrics["sd"] = standard_deviation(texture).item()

    return metrics


def run_evaluation(config: dict) -> None:
    """Run evaluation comparing SLOT vs HADAR on synthetic data.

    Args:
        config: loaded TOML configuration dict.
    """
    slot_cfg = config.get("slot", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.get("training", {}).get("seed", 42))

    # Output
    report_dir = Path(f"{ARTIFACTS}/reports/{PROJECT}")
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset (use fewer scenes for eval)
    n_eval = data_cfg.get("n_eval_scenes", 20)
    dataset = SyntheticThermalDataset(
        n_scenes=n_eval,
        height=data_cfg.get("height", 64),
        width=data_cfg.get("width", 64),
        nedt=data_cfg.get("nedt", 0.03),
        seed=config.get("training", {}).get("seed", 42) + 1000,  # different from train
    )

    wg = default_wavenumber_grid(device)

    # Models
    slot = SLOTDecomposer(
        n_knots=slot_cfg.get("n_knots", 20),
        reg_lambda=slot_cfg.get("reg_lambda", 1.0),
        max_iter=slot_cfg.get("max_iter", 100),
        wavenumber_grid=wg,
    ).to(device)

    hadar = HADARDecomposer(wavenumber_grid=wg).to(device)

    slot_metrics_all = []
    hadar_metrics_all = []

    for idx in range(len(dataset)):
        scene = dataset[idx]
        s_obs = scene["s_obs"].to(device)
        s_sky = scene["s_sky"].to(device)
        s_ground = scene["s_ground"].to(device)
        t_gt = scene["t_gt"].to(device)
        e_gt = scene["e_gt"].to(device)
        v_gt = scene["v_gt"].to(device)

        # SLOT decomposition
        slot_result = slot.decompose(s_obs, s_sky, s_ground)
        slot_m = evaluate_decomposition(
            slot_result.temperature, slot_result.emissivity,
            slot_result.view_factor, slot_result.texture,
            t_gt, e_gt, v_gt,
        )
        slot_metrics_all.append(slot_m)

        # HADAR decomposition
        s_obs_flat = s_obs.reshape(-1, s_obs.shape[-1])
        hadar_result = hadar.decompose(s_obs_flat, s_sky, s_ground)

        h, w = s_obs.shape[:2]
        hadar_m = evaluate_decomposition(
            hadar_result.temperature.reshape(h, w),
            hadar_result.emissivity.reshape(h, w, -1),
            hadar_result.view_factor.reshape(h, w),
            hadar_result.texture.reshape(h, w),
            t_gt, e_gt, v_gt,
        )
        hadar_metrics_all.append(hadar_m)

        if idx % 5 == 0:
            print(
                f"[Eval {idx:3d}/{n_eval}] "
                f"SLOT T_RMSE={slot_m['t_rmse']:.2f}K "
                f"HADAR T_RMSE={hadar_m['t_rmse']:.2f}K"
            )

    # Aggregate metrics
    def avg_metrics(metrics_list: list[dict]) -> dict[str, float]:
        keys = metrics_list[0].keys()
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

    slot_avg = avg_metrics(slot_metrics_all)
    hadar_avg = avg_metrics(hadar_metrics_all)

    # Report
    report = {
        "run_id": run_id,
        "n_scenes": n_eval,
        "slot": slot_avg,
        "hadar": hadar_avg,
        "config": config,
    }

    report_path = report_dir / f"eval_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("EVALUATION REPORT: SLOT vs HADAR")
    print("=" * 70)
    print(f"{'Metric':<20} {'SLOT':>12} {'HADAR':>12} {'Winner':>10}")
    print("-" * 70)
    for key in slot_avg:
        sv = slot_avg[key]
        hv = hadar_avg[key]
        # Lower is better for error metrics, higher for quality metrics
        if key in ("en", "ag", "sf", "sd"):
            winner = "SLOT" if sv > hv else "HADAR"
        else:
            winner = "SLOT" if sv < hv else "HADAR"
        print(f"{key:<20} {sv:>12.4f} {hv:>12.4f} {winner:>10}")
    print("=" * 70)
    print(f"Report saved: {report_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TAG evaluation pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    print("[TAG] Evaluation Pipeline")
    print(f"[CONFIG] {args.config}")

    run_evaluation(config)


if __name__ == "__main__":
    main()
