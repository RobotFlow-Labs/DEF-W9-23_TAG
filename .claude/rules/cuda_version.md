# Rule: CUDA Version — ALWAYS cu128

## RULE
ALWAYS install PyTorch with CUDA 12.8. NEVER use default torch (ships CUDA 13 which is NOT compatible with this server).

## Install Command
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Why
- Server has CUDA 12.x system-wide
- Default pip torch bundles CUDA 13 (not mature, breaks CUDA extensions)
- CUDA extensions compiled against system CUDA 12 will NOT work with torch CUDA 13
- Multiple agents have wasted hours debugging this mismatch

## DO NOT
- DO NOT run `uv pip install torch` without `--index-url`
- DO NOT use `cu130` or `cu131` wheels
- DO NOT ignore CUDA version mismatch errors — fix by reinstalling torch cu128

## In pyproject.toml
```toml
[tool.uv]
extra-index-url = ["https://download.pytorch.org/whl/cu128"]
```
