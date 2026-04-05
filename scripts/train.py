#!/usr/bin/env python
"""CLI script to run TAG SLOT optimization pipeline.

Usage:
    uv run python scripts/train.py --config configs/default.toml
    uv run python scripts/train.py --config configs/debug.toml --max-scenes 5
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tag.train import main

if __name__ == "__main__":
    main()
