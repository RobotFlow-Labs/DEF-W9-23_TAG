# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 65%

## Done
- [x] Read paper (arXiv 2604.01542) — TAG thermal anti-ghosting
- [x] Created CLAUDE.md with paper summary, architecture, hyperparameters
- [x] Created ASSETS.md with asset inventory
- [x] Created PRD.md with 7-PRD build plan
- [x] Created 7 PRD files (PRD-01 through PRD-07)
- [x] Created tasks/INDEX.md with granular tasks
- [x] Created pyproject.toml with hatchling backend
- [x] Created TOML configs (default, paper, debug)
- [x] PRD-01: Foundation verified — Planck, B-spline, forward model, 37 tests pass
- [x] PRD-02: CUDA-accelerated B-spline + Planck kernels (torch.compile, vectorized)
- [x] PRD-03: Real thermal dataset adapters (VIVID++ 71K images, DroneVehicle-night)
- [x] PRD-04: CUDA training pipeline with multi-dataset support
- [x] PRD-06: Export pipeline (ONNX verified, TRT FP16/FP32 ready)
- [x] 53/53 tests pass (model, dataset, CUDA, export)

## In Progress
- [ ] PRD-05: Run evaluation on synthetic + real thermal data
- [ ] PRD-07: Docker + ROS2 integration

## TODO
- [ ] Run multi-dataset training: synthetic → VIVID++ → DroneVehicle → combined
- [ ] ONNX + TRT export with real optimized parameters
- [ ] Copy CUDA kernels to /mnt/forge-data/shared_infra/cuda_extensions/
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_tag
- [ ] Docker build and health check
- [ ] Git push to remote

## Blocking
- None — all code paths verified, datasets on disk

## Downloads Needed
- None — all datasets available locally

## Available Datasets
- Synthetic (generated at runtime, controlled ground truth)
- VIVID++ thermal: /mnt/train-data/datasets/vivid_plus_plus/ (47GB, 71K 16-bit thermal images)
- DroneVehicle-night: /mnt/forge-data/datasets/wave9/drones/DroneVehicle-night.zip (needs unzip)
- nuScenes: /mnt/forge-data/datasets/nuscenes/ (479GB, night scenes)
- NUAA-SIRST: /mnt/forge-data/datasets/nuaa_sirst_yolo/ (IR small targets)
