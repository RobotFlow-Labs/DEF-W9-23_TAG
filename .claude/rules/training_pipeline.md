# ANIMA Training Pipeline — MANDATORY Rules

## Data Split
- Train: 90% | Val: 5% | Test: 5%
- Save split indices to `data_split.json`
- Test set is ONLY used for final evaluation

## Early Stopping
- Patience: 10 epochs without val_loss improvement
- Min delta: 1e-4
- Plateau LR: reduce by 0.5 after 5 stagnant epochs

## Checkpointing
- Keep ONLY last 2 best checkpoints (by val_loss)
- Delete older checkpoints automatically
- Naming:
  ```
  project_loki_cuda_v{version}_epoch{N}_val{loss:.4f}.pth
  project_loki_mlx_v{version}_epoch{N}_val{loss:.4f}.pth
  ```
- Save `training_config.json` alongside each checkpoint

## Learning Rate
- Cosine annealing with 5% linear warmup
- Min LR: 1e-6

## Training Settings
- bf16 mixed precision on CUDA
- fp32 on MLX (Apple Silicon)
- Gradient clipping: max_norm=1.0
- Seed: 42 (torch, numpy, random) — save in checkpoint
- Every script MUST support `--resume path/to/checkpoint.pth`

## Logging
- TensorBoard events in `tensorboard/` dir
- JSON history in `logs/training_history.json`
- Console: `[Epoch N/total] train_loss=X val_loss=Y lr=Z`

## After Training
1. Evaluate on TEST split (not val)
2. Generate `TRAINING_REPORT.md` with config table, loss curves, metrics
3. Export best model to ONNX
4. Update NEXT_STEPS.md
5. Push to HF: `hf upload ilessio-aiflowlab/PROJECT_NAME . . --private`
