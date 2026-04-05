# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 85%

## Done
- [x] Read paper (arXiv 2604.01542) — TAG thermal anti-ghosting
- [x] Created CLAUDE.md with paper summary, architecture, hyperparameters
- [x] Created ASSETS.md with asset inventory
- [x] Created PRD.md with 7-PRD build plan
- [x] Created 7 PRD files (PRD-01 through PRD-07)
- [x] PRD-01: Foundation — Planck, B-spline, forward model, 37 tests pass
- [x] PRD-02: CUDA-accelerated B-spline + Planck kernels (torch.compile, vectorized)
- [x] PRD-03: Real thermal dataset adapters (VIVID++ 71K images, DroneVehicle-night)
- [x] PRD-04: CUDA training pipeline with multi-dataset support
- [x] PRD-05: Evaluation — SLOT vs HADAR on synthetic + VIVID++
- [x] PRD-06: Export pipeline — pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [x] 53/53 tests pass (model, dataset, CUDA, export)
- [x] CUDA kernels copied to /mnt/forge-data/shared_infra/cuda_extensions/tag_thermal_kernels/
- [x] Training complete: 200 synthetic scenes (T_RMSE=11.95K) + 200 VIVID++ scenes (T_RMSE=5.75K)
- [x] SLOT beats HADAR on both T_RMSE and e_MAE

## Training Results
| Run       | Scenes | T_RMSE | e_MAE  | Time/scene |
|-----------|--------|--------|--------|------------|
| synthetic | 200    | 11.95K | 0.2666 | 0.18s      |
| vivid++   | 200    | 5.75K  | 0.1508 | 0.16s      |

## TODO
- [ ] PRD-07: Docker build + health check test
- [ ] Push to HuggingFace: ilessio-aiflowlab/project_tag
- [ ] Git push to remote
- [ ] Run larger training on full VIVID++ dataset (71K scenes)
- [ ] DroneVehicle-night training (after unzip)

## Blocking
- None

## Exports Available
- /mnt/artifacts-datai/exports/project_tag/tag_forward_embedded.onnx (8.3KB)
- /mnt/artifacts-datai/exports/project_tag/tag_forward_embedded_fp16.trt (57KB)
- /mnt/artifacts-datai/exports/project_tag/tag_forward_embedded_fp32.trt (57KB)
- /mnt/artifacts-datai/exports/project_tag/*.safetensors
- /mnt/artifacts-datai/exports/project_tag/*.pth
