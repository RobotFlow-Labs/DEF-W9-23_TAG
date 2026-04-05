# Rule: GPU Readiness Check — MANDATORY Before Training

## RULE
Before ANY training run, you MUST:
1. Ask the user: "Are the GPUs available for training? Which GPU IDs can I use?"
2. Wait for confirmation — DO NOT assume GPUs are free
3. Run /gpu-batch-finder on the confirmed GPU to auto-detect optimal batch size
4. Only THEN start training with the detected batch size

## WHY
We share GPUs across 90+ modules. Multiple agents may be running. Starting training without checking wastes GPU hours if another module is already using it. The batch finder needs an empty GPU to measure correctly.

## HOW
```
Agent: "GPUs needed for training. Are any available? Which GPU ID should I use?"
User: "Use GPU 3"
Agent: runs /gpu-batch-finder on cuda:3
Agent: "Optimal batch size: 288 (63% of 23GB). Starting training with nohup + disown."
```

## DO NOT
- DO NOT start training without asking the user first
- DO NOT assume any GPU is free
- DO NOT skip /gpu-batch-finder — paper batch sizes assume different hardware
- DO NOT use batch sizes from the paper without verifying on our GPUs
