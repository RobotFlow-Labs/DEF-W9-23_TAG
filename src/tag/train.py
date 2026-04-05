"""Training (optimization) pipeline for TAG SLOT decomposition.

TAG "training" is per-scene optimization, not gradient-based NN training.
This module runs SLOT decomposition on a dataset of thermal scenes,
logs convergence, and saves results.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from tag.dataset import SyntheticThermalDataset
from tag.model import SLOTDecomposer
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


def setup_output_dirs() -> dict[str, Path]:
    """Create output directories under artifacts."""
    dirs = {
        "checkpoints": Path(f"{ARTIFACTS}/checkpoints/{PROJECT}"),
        "logs": Path(f"{ARTIFACTS}/logs/{PROJECT}"),
        "reports": Path(f"{ARTIFACTS}/reports/{PROJECT}"),
        "tensorboard": Path(f"{ARTIFACTS}/tensorboard/{PROJECT}"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_optimization(config: dict) -> None:
    """Run SLOT optimization on synthetic dataset.

    Args:
        config: loaded TOML configuration dict.
    """
    # Extract config sections
    slot_cfg = config.get("slot", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.get("training", {}).get("seed", 42))

    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] SLOT: n_knots={slot_cfg.get('n_knots', 20)}, "
          f"lambda={slot_cfg.get('reg_lambda', 1.0)}, "
          f"max_iter={slot_cfg.get('max_iter', 100)}")

    # Setup output
    dirs = setup_output_dirs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dirs["logs"] / f"train_{run_id}.jsonl"

    # Create dataset
    dataset = SyntheticThermalDataset(
        n_scenes=data_cfg.get("n_scenes", 100),
        height=data_cfg.get("height", 64),
        width=data_cfg.get("width", 64),
        n_materials=data_cfg.get("n_materials", 5),
        t_range=tuple(data_cfg.get("t_range", [270.0, 330.0])),
        nedt=data_cfg.get("nedt", 0.03),
        seed=config.get("training", {}).get("seed", 42),
    )

    print(f"[DATA] {len(dataset)} synthetic scenes, "
          f"{data_cfg.get('height', 64)}x{data_cfg.get('width', 64)} pixels")

    # Create SLOT decomposer
    wg = default_wavenumber_grid(device)
    decomposer = SLOTDecomposer(
        n_knots=slot_cfg.get("n_knots", 20),
        reg_lambda=slot_cfg.get("reg_lambda", 1.0),
        max_iter=slot_cfg.get("max_iter", 100),
        tolerance=slot_cfg.get("tolerance", 1e-6),
        lr=slot_cfg.get("lr", 0.01),
        constraint_mu=slot_cfg.get("constraint_mu", 10.0),
        wavenumber_grid=wg,
    ).to(device)

    # Run optimization on each scene
    results = []
    total_time = 0.0

    with open(log_path, "w") as log_file:
        for scene_idx in range(len(dataset)):
            scene = dataset[scene_idx]
            s_obs = scene["s_obs"].to(device)
            s_sky = scene["s_sky"].to(device)
            s_ground = scene["s_ground"].to(device)

            t_start = time.time()
            result = decomposer.decompose(
                s_obs, s_sky, s_ground,
                verbose=(scene_idx == 0),  # verbose on first scene only
            )
            elapsed = time.time() - t_start
            total_time += elapsed

            # Compute reconstruction error
            t_gt = scene["t_gt"].to(device)
            t_rmse = torch.sqrt(((result.temperature - t_gt) ** 2).mean()).item()

            e_gt = scene["e_gt"].to(device)
            e_mae = (result.emissivity - e_gt).abs().mean().item()

            log_entry = {
                "scene": scene_idx,
                "objective": result.objective,
                "n_iterations": result.n_iterations,
                "t_rmse": t_rmse,
                "e_mae": e_mae,
                "time_s": elapsed,
            }
            log_file.write(json.dumps(log_entry) + "\n")

            results.append(log_entry)

            if scene_idx % 10 == 0:
                print(
                    f"[Scene {scene_idx:4d}/{len(dataset)}] "
                    f"obj={result.objective:.6f} "
                    f"T_RMSE={t_rmse:.2f}K "
                    f"e_MAE={e_mae:.4f} "
                    f"iters={result.n_iterations} "
                    f"time={elapsed:.2f}s"
                )

            # Save checkpoint periodically
            save_every = output_cfg.get("save_every_n_scenes", 50)
            if (scene_idx + 1) % save_every == 0:
                ckpt_path = dirs["checkpoints"] / f"results_scene{scene_idx:04d}.pt"
                torch.save(
                    {
                        "scene_idx": scene_idx,
                        "temperature": result.temperature.cpu(),
                        "emissivity": result.emissivity.cpu(),
                        "view_factor": result.view_factor.cpu(),
                        "config": config,
                    },
                    ckpt_path,
                )
                print(f"  Saved checkpoint: {ckpt_path}")

    # Summary
    avg_t_rmse = sum(r["t_rmse"] for r in results) / len(results)
    avg_e_mae = sum(r["e_mae"] for r in results) / len(results)
    avg_time = total_time / len(results)

    print("\n" + "=" * 60)
    print(f"[DONE] {len(results)} scenes processed in {total_time:.1f}s")
    print(f"  Avg T_RMSE: {avg_t_rmse:.2f} K")
    print(f"  Avg e_MAE:  {avg_e_mae:.4f}")
    print(f"  Avg time:   {avg_time:.2f} s/scene")
    print(f"  Logs: {log_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TAG SLOT optimization pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Override number of scenes to process",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.max_scenes is not None:
        config.setdefault("data", {})["n_scenes"] = args.max_scenes

    print("[TAG] SLOT Optimization Pipeline")
    print(f"[CONFIG] {args.config}")

    run_optimization(config)


if __name__ == "__main__":
    main()
