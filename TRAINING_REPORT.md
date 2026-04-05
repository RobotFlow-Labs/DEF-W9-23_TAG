# TAG Training Report
> Module: project_tag (Thermal Anti-Ghosting)
> Paper: "Universal computational thermal imaging overcoming the ghosting effect" (arXiv 2604.01542)
> Date: 2026-04-05

## Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | SLOT (Smoothness-structured Library-free Optimization for TeX) |
| B-spline knots | 20 (22 basis functions) |
| Regularization λ | 1.0 |
| Max iterations | 200 (paper config) |
| Tolerance | 1e-8 |
| Constraint μ | 50.0 |
| Spectral range | 870-1269 cm⁻¹ (67 bands @ 6 cm⁻¹) |
| GPU | NVIDIA L4 (23GB VRAM) |
| VRAM used | ~12GB (52%) |
| Backend | PyTorch 2.11 + torch.compile (Triton) |

## Results

### Run 1: Synthetic Data (200 scenes, 256×320 pixels)
| Metric | Value |
|--------|-------|
| T_RMSE | 11.95 K |
| e_MAE | 0.2666 |
| Avg time/scene | 0.18s |
| Total time | 36.9s |

### Run 2: VIVID++ Thermal (200 scenes, 256×320 pixels)
| Metric | Value |
|--------|-------|
| T_RMSE | 5.75 K |
| e_MAE | 0.1508 |
| Avg time/scene | 0.16s |
| Total time | 32.9s |

### SLOT vs HADAR Comparison (20 synthetic scenes)
| Metric | SLOT | HADAR | Winner |
|--------|------|-------|--------|
| T_RMSE | 10.81 K | 11.82 K | SLOT |
| e_MAE | 0.2839 | 0.3269 | SLOT |

## CUDA Acceleration
- Vectorized B-spline basis (no recursion): ~10x speedup vs recursive
- torch.compile fused Planck + rendering: eliminates intermediate allocations
- GPU Voronoi region assignment: O(HW) vs O(HW·N) nested loops
- Chunked processing: 4096 pixels/chunk for memory control

## Exports
| Format | File | Size |
|--------|------|------|
| PyTorch (.pth) | tag_forward_*.pth | 9.8 KB |
| safetensors | tag_forward_*.safetensors | 6.3 KB |
| ONNX | tag_forward_embedded.onnx | 8.3 KB |
| TensorRT FP16 | tag_forward_embedded_fp16.trt | 57 KB |
| TensorRT FP32 | tag_forward_embedded_fp32.trt | 57 KB |

## Hardware
- GPU: NVIDIA L4 (23GB VRAM, sm_89)
- CUDA: 12.8 (torch cu128)
- Python: 3.11
- PyTorch: 2.11.0+cu128
