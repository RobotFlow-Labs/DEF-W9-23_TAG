# GPU MEMORY RULE — MANDATORY

EVERY training run MUST use at least 60% of GPU VRAM. Target 70-80%.

After launching training, CHECK within 60 seconds:
  nvidia-smi -i $GPU_ID --query-gpu=memory.used,memory.total --format=csv,noheader

If memory.used < 60% of memory.total:
1. STOP training immediately
2. DOUBLE the batch_size
3. Restart and check again

L4 = 23GB. Minimum: 14GB used. Target: 16-18GB.
DO NOT run training at <30% VRAM.
