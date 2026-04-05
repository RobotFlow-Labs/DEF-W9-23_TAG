#!/usr/bin/env python
"""CLI wrapper for CUDA-accelerated TAG training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tag.train_cuda import main

if __name__ == "__main__":
    main()
