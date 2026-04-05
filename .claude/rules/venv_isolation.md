# Rule: Pure venv Isolation — MANDATORY

## RULE
Every module MUST use its own isolated .venv inside the project folder.
NEVER use system Python. NEVER use PYTHONPATH to share packages across modules.
NEVER install packages globally with --user or --break-system-packages.

## WHY
Modules have conflicting dependencies (torch versions, mmcv versions, etc).
Shared PYTHONPATH = one module breaks another. Pure isolation = zero conflicts.

## HOW

### Create venv (FIRST thing when entering a module)
```bash
cd /mnt/forge-data/modules/project_{module}
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync
```

### Verify isolation
```bash
which python  # MUST show .venv/bin/python, NOT /usr/bin/python
pip list      # MUST only show this module deps
```

### In training scripts
```bash
# CORRECT — uses module venv
cd /mnt/forge-data/modules/project_{module}
source .venv/bin/activate
python scripts/train.py

# ALSO CORRECT — uv run handles venv automatically
cd /mnt/forge-data/modules/project_{module}
uv run python scripts/train.py

# WRONG — system python
python3 scripts/train.py

# WRONG — PYTHONPATH sharing
PYTHONPATH=/mnt/forge-data/modules/project_other/src python scripts/train.py
```

### In Dockerfiles
```dockerfile
# Already isolated by container — no venv needed inside Docker
```

### .venv location
The .venv MUST be inside the project folder, NOT in a shared location.
UV caches are at /mnt/forge-data/.uv-cache/ (symlinked from ~/.cache/uv/).
This means package wheels are cached globally but venvs are per-module.

## DO NOT
- DO NOT use `pip install --user` or `pip install --break-system-packages`
- DO NOT set PYTHONPATH to point to other modules
- DO NOT share .venv across modules
- DO NOT use system Python for any module work
- DO NOT skip `uv venv` — create it first thing when entering a module
