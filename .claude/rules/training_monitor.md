# Training Monitor — MANDATORY

## Rule: Check GPU availability before training

Before starting ANY GPU training run, execute:
```bash
python3 /mnt/forge-data/scripts/training_monitor.py
```

This shows which GPUs are free and which modules are currently training.
DO NOT start training on a GPU that is already in use.

During training, use `--watch` mode to monitor progress:
```bash
python3 /mnt/forge-data/scripts/training_monitor.py project_loki --watch
```
