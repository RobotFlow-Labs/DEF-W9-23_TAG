# PRD-06: Export Pipeline

> Status: TODO
> Module: TAG (Thermal Anti-Ghosting)
> Depends on: PRD-02, PRD-05

## Objective

Export TAG components for deployment: batched Planck function, B-spline evaluation,
and the full SLOT forward pass as ONNX/TensorRT modules.

## Deliverables

### 1. ONNX Export

Export the forward model (Planck + rendering equation) as ONNX:
```python
# Input: T [N], beta [N, K], V [N], wavenumber_grid [C]
# Output: S_model [N, C]
torch.onnx.export(forward_model, dummy_input, "tag_forward.onnx")
```

### 2. TensorRT Optimization

Use shared TRT toolkit:
```bash
python /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py \
  --onnx /mnt/artifacts-datai/exports/project_tag/tag_forward.onnx \
  --output /mnt/artifacts-datai/exports/project_tag/ \
  --fp16 --fp32
```

### 3. Safetensors Export

Save optimized B-spline basis and default parameters as safetensors.

### 4. Export to HuggingFace

```bash
huggingface-cli upload ilessio-aiflowlab/project_tag-checkpoint \
  /mnt/artifacts-datai/exports/project_tag/ .
```

## Acceptance Criteria
- [ ] ONNX model produces identical output to PyTorch (rtol=1e-5)
- [ ] TRT FP16 and FP32 engines built successfully
- [ ] Safetensors file loadable
- [ ] HF upload succeeds
