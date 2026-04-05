# Dual Backend — CUDA + MLX Coexistence

## CRITICAL: Never remove MLX code. CUDA must be PARALLEL.

### File Naming Convention
```
*_cuda_*     — CUDA/PyTorch backend (GPU server, Jetson)
*_mlx_*      — MLX backend (Apple Silicon)
*_onnx_*     — ONNX export
*_trt_*      — TensorRT export
*_coreml_*   — CoreML export
```

### Backend Structure
```
src/project_loki/
├── backends/
│   ├── __init__.py      # Auto-detect: CUDA > MLX > CPU
│   ├── cuda/            # CUDA implementation
│   └── mlx/             # MLX implementation (DO NOT TOUCH)
├── train.py             # Entry point — selects backend automatically
└── export.py            # ONNX/TRT/CoreML export
```

### Rules
1. NEVER delete or modify existing MLX code
2. Add CUDA backends alongside MLX
3. Both backends must produce identical output format
4. Use `--backend cuda|mlx|auto` CLI flag
5. Default: auto-detect (CUDA if available, else MLX, else CPU)
