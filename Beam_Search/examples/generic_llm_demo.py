"""Model-agnostic raw-token demo for merged ONNX beam graphs."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLKIT_DIR = Path(__file__).resolve().parent.parent
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from onnx_beam_search.cli_run import main


if __name__ == "__main__":
    main()