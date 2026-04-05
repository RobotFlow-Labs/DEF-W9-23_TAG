# Rule: venv Startup — FIRST THING ON BOOT

## RULE
When you enter this module, the VERY FIRST command you run is:

```bash
cd /mnt/forge-data/modules/{this_module}
source .venv/bin/activate
which python  # MUST show .venv/bin/python
```

If .venv does not exist, create it:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv sync --no-dev 2>/dev/null || uv pip install numpy opencv-python pillow scipy tqdm tensorboard safetensors onnx transformers accelerate huggingface-hub ruff pytest
```

## NEVER DO THIS
```bash
# WRONG — uses system python, breaks other modules
python3 scripts/train.py
pip install something
pip install --user something
pip install --break-system-packages something
export PYTHONPATH=/some/other/module
```

## ALWAYS DO THIS
```bash
# RIGHT — uses isolated venv
source .venv/bin/activate
python scripts/train.py
# OR
uv run python scripts/train.py
```

## WHY
93 modules share this server. System Python = dependency hell. One module installs mmcv 1.7, another needs mmcv 2.1 — both break. Pure venv isolation = zero conflicts.

Root disk filled up (290GB) because agents installed packages globally. NEVER AGAIN.

## VERIFICATION
After activating, verify:
```bash
which python    # /mnt/forge-data/modules/project_X/.venv/bin/python ← CORRECT
which python    # /usr/bin/python3 ← WRONG, stop and activate venv
```
