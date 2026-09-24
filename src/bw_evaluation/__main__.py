"""Allow `python -m bw_evaluation FOLDER`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
