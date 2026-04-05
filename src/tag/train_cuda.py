"""CUDA-accelerated training pipeline for TAG SLOT decomposition.

Supports multi-dataset training:
  Run 1: Synthetic data (controlled ground truth)
  Run 2: VIVID++ thermal (real 16-bit thermal, 320x256)
  Run 3: DroneVehicle-night (drone IR)
  Run 4: Combined (all sources)

Each "training run" is SLOT optimization over a dataset — not NN training.
Results are checkpointed per scene with metrics logged to JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tag.cuda.slot_cuda import SLOTDecomposerCUDA
from tag.dataset import SyntheticThermalDataset
from tag.dataset_thermal import (
    CombinedThermalDataset,
    DroneVehicleNightDataset,
    VIVIDThermalDataset,
)
from tag.losses import (
    average_gradient,
    information_entropy,
    spatial_frequency,
    spectral_angle_mapper,
    standard_deviation,
)
from tag.utils import default_wavenumber_grid, seed_everything

PROJECT = "project_tag"
ARTIFACTS = "/mnt/artifacts-datai"


def load_config(config_path: str) -> dict:
    """Load TOML configuration file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def setup_output_dirs(run_name: str = "") -> dict[str, Path]:
    """Create output directories."""
    suffix = f"/{run_name}" if run_name else ""
    dirs = {
        "checkpoints": Path(f"{ARTIFACTS}/checkpoints/{PROJECT}{suffix}"),
        "logs": Path(f"{ARTIFACTS}/logs/{PROJECT}"),
        "reports": Path(f"{ARTIFACTS}/reports/{PROJECT}"),
        "tensorboard": Path(f"{ARTIFACTS}/tensorboard/{PROJECT}"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_metrics(
    result_t: torch.Tensor,
    result_e: torch.Tensor,
    result_v: torch.Tensor,
    result_texture: torch.Tensor,
    gt_t: torch.Tensor,
    gt_e: torch.Tensor,
    gt_v: torch.Tensor,
) -> dict[str, float]:
    """Compute all metrics for a scene."""
    m = {}
    m["t_rmse"] = torch.sqrt(((result_t - gt_t) ** 2).mean()).item()
    m["t_mae"] = (result_t - gt_t).abs().mean().item()
    m["e_mae"] = (result_e - gt_e).abs().mean().item()
    m["e_rmse"] = torch.sqrt(((result_e - gt_e) ** 2).mean()).item()
    m["v_rmse"] = torch.sqrt(((result_v - gt_v) ** 2).mean()).item()

    # Spectral angle mapper
    pred_flat = result_e.reshape(-1, result_e.shape[-1])
    gt_flat = gt_e.reshape(-1, gt_e.shape[-1])
    m["sam_rad"] = spectral_angle_mapper(pred_flat, gt_flat).item()

    # Image quality metrics on texture
    if result_texture.dim() == 2:
        m["en"] = information_entropy(result_texture).item()
        m["ag"] = average_gradient(result_texture).item()
        m["sf"] = spatial_frequency(result_texture).item()
        m["sd"] = standard_deviation(result_texture).item()

    return m


def run_single_dataset(
    decomposer: SLOTDecomposerCUDA,
    dataset,
    device: torch.device,
    run_name: str,
    config: dict,
    verbose: bool = True,
) -> list[dict]:
    """Run SLOT decomposition on an entire dataset.

    Args:
        decomposer: CUDA SLOT decomposer.
        dataset: thermal dataset.
        device: GPU device.
        run_name: name for this run (e.g., 'synthetic', 'vivid').
        config: full configuration dict.
        verbose: print progress.

    Returns:
        List of per-scene metric dicts.
    """
    dirs = setup_output_dirs(run_name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dirs["logs"] / f"train_{run_name}_{run_id}.jsonl"

    output_cfg = config.get("output", {})
    save_every = output_cfg.get("save_every_n_scenes", 50)

    results = []
    total_time = 0.0
    n_scenes = len(dataset)

    print(f"\n{'=' * 60}")
    print(f"[RUN] {run_name} — {n_scenes} scenes")
    print(f"[DEVICE] {device}")
    print(f"[LOG] {log_path}")
    print(f"{'=' * 60}")

    with open(log_path, "w") as log_file:
        for scene_idx in range(n_scenes):
            scene = dataset[scene_idx]
            s_obs = scene["s_obs"].to(device)
            s_sky = scene["s_sky"].to(device)
            s_ground = scene["s_ground"].to(device)

            t_start = time.time()
            result = decomposer.decompose(
                s_obs, s_sky, s_ground,
                verbose=(verbose and scene_idx == 0),
            )
            elapsed = time.time() - t_start
            total_time += elapsed

            # Metrics
            t_gt = scene["t_gt"].to(device)
            e_gt = scene["e_gt"].to(device)
            v_gt = scene["v_gt"].to(device)

            metrics = compute_metrics(
                result.temperature, result.emissivity,
                result.view_factor, result.texture,
                t_gt, e_gt, v_gt,
            )

            log_entry = {
                "run": run_name,
                "scene": scene_idx,
                "objective": result.objective,
                "n_iterations": result.n_iterations,
                "time_s": elapsed,
                **metrics,
            }
            if "path" in scene:
                log_entry["path"] = scene["path"]

            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            results.append(log_entry)

            if verbose and scene_idx % max(1, n_scenes // 20) == 0:
                print(
                    f"  [{run_name}] Scene {scene_idx:5d}/{n_scenes} "
                    f"obj={result.objective:.6f} "
                    f"T_RMSE={metrics['t_rmse']:.2f}K "
                    f"e_MAE={metrics['e_mae']:.4f} "
                    f"time={elapsed:.2f}s"
                )

            # Checkpoint
            if (scene_idx + 1) % save_every == 0:
                ckpt_path = dirs["checkpoints"] / f"result_{run_name}_s{scene_idx:05d}.pt"
                torch.save({
                    "scene_idx": scene_idx,
                    "temperature": result.temperature.cpu(),
                    "emissivity": result.emissivity.cpu(),
                    "view_factor": result.view_factor.cpu(),
                    "texture": result.texture.cpu(),
                    "config": config,
                    "run": run_name,
                }, ckpt_path)

    # Summary
    avg_metrics = {}
    for key in results[0]:
        if isinstance(results[0][key], (int, float)) and key not in ("scene",):
            vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
            if vals:
                avg_metrics[key] = sum(vals) / len(vals)

    print(f"\n[DONE] {run_name}: {n_scenes} scenes in {total_time:.1f}s")
    print(f"  Avg T_RMSE: {avg_metrics.get('t_rmse', 0):.2f} K")
    print(f"  Avg e_MAE:  {avg_metrics.get('e_mae', 0):.4f}")
    print(f"  Avg EN:     {avg_metrics.get('en', 0):.2f}")
    print(f"  Avg time:   {total_time / n_scenes:.2f} s/scene")

    # Save summary
    summary_path = dirs["reports"] / f"summary_{run_name}_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump({"run": run_name, "n_scenes": n_scenes,
                    "total_time_s": total_time, "avg_metrics": avg_metrics}, f, indent=2)
    print(f"  Summary: {summary_path}")

    return results


def main():
    """CLI entry point for CUDA-accelerated multi-dataset training."""
    parser = argparse.ArgumentParser(description="TAG CUDA-accelerated training pipeline")
    parser.add_argument("--config", type=str, default="configs/paper.toml")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--runs",
        type=str,
        default="synthetic",
        help="Comma-separated: synthetic,vivid,drone,combined",
    )
    parser.add_argument("--vivid-dir", type=str,
                        default="/mnt/train-data/datasets/vivid_plus_plus/dataset/train")
    parser.add_argument("--drone-dir", type=str,
                        default="/mnt/forge-data/datasets/wave9/drones/DroneVehicle-night")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_everything(config.get("training", {}).get("seed", 42))

    slot_cfg = config.get("slot", {})
    data_cfg = config.get("data", {})

    print("[TAG] CUDA-Accelerated Multi-Dataset Training Pipeline")
    print(f"[CONFIG] {args.config}")
    print(f"[DEVICE] {device}")
    print(f"[RUNS] {args.runs}")

    # Create CUDA decomposer
    wg = default_wavenumber_grid(device)
    decomposer = SLOTDecomposerCUDA(
        n_knots=slot_cfg.get("n_knots", 20),
        reg_lambda=slot_cfg.get("reg_lambda", 1.0),
        max_iter=slot_cfg.get("max_iter", 100),
        tolerance=slot_cfg.get("tolerance", 1e-6),
        lr=slot_cfg.get("lr", 0.01),
        constraint_mu=slot_cfg.get("constraint_mu", 10.0),
        chunk_size=args.chunk_size,
        wavenumber_grid=wg,
    ).to(device)

    runs = args.runs.split(",")
    all_results = {}

    for run_name in runs:
        run_name = run_name.strip()

        if run_name == "synthetic":
            n_scenes = args.max_scenes or data_cfg.get("n_scenes", 100)
            dataset = SyntheticThermalDataset(
                n_scenes=n_scenes,
                height=data_cfg.get("height", 64),
                width=data_cfg.get("width", 64),
                n_materials=data_cfg.get("n_materials", 5),
                t_range=tuple(data_cfg.get("t_range", [270.0, 330.0])),
                nedt=data_cfg.get("nedt", 0.03),
                seed=config.get("training", {}).get("seed", 42),
            )

        elif run_name == "vivid":
            dataset = VIVIDThermalDataset(
                root_dir=args.vivid_dir,
                split="train",
                max_scenes=args.max_scenes or 1000,
                resize=(data_cfg.get("height", 256), data_cfg.get("width", 320)),
                wavenumber_grid=wg.cpu(),
            )

        elif run_name == "drone":
            dataset = DroneVehicleNightDataset(
                root_dir=args.drone_dir,
                split="train",
                max_scenes=args.max_scenes or 500,
                target_size=(data_cfg.get("height", 256), data_cfg.get("width", 320)),
                wavenumber_grid=wg.cpu(),
            )

        elif run_name == "combined":
            synth_ds = SyntheticThermalDataset(
                n_scenes=args.max_scenes or 50,
                height=data_cfg.get("height", 64),
                width=data_cfg.get("width", 64),
            )
            vivid_ds = VIVIDThermalDataset(
                root_dir=args.vivid_dir,
                split="train",
                max_scenes=args.max_scenes or 200,
                resize=(data_cfg.get("height", 256), data_cfg.get("width", 320)),
                wavenumber_grid=wg.cpu(),
            )
            dataset = CombinedThermalDataset([synth_ds, vivid_ds])

        else:
            print(f"[WARN] Unknown run: {run_name}, skipping")
            continue

        print(f"\n[DATASET] {run_name}: {len(dataset)} scenes")
        results = run_single_dataset(
            decomposer, dataset, device, run_name, config,
        )
        all_results[run_name] = results

    # Final comparison table
    print("\n" + "=" * 70)
    print("MULTI-DATASET COMPARISON")
    print("=" * 70)
    print(f"{'Run':<15} {'Scenes':>8} {'T_RMSE':>10} {'e_MAE':>10} {'EN':>8} {'AG':>8}")
    print("-" * 70)
    for rn, results in all_results.items():
        n = len(results)
        t_rmse = sum(r.get("t_rmse", 0) for r in results) / n
        e_mae = sum(r.get("e_mae", 0) for r in results) / n
        en = sum(r.get("en", 0) for r in results) / n
        ag = sum(r.get("ag", 0) for r in results) / n
        print(f"{rn:<15} {n:>8} {t_rmse:>10.2f} {e_mae:>10.4f} {en:>8.2f} {ag:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
