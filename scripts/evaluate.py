#!/usr/bin/env python
"""CLI script to run TAG evaluation pipeline.

Usage:
    uv run python scripts/evaluate.py --config configs/default.toml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tag.evaluate import main

if __name__ == "__main__":
    main()
