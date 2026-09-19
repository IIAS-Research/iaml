"""Allow running the test suite via `python -m tests`."""
from __future__ import annotations

import sys

from .tests import main


if __name__ == "__main__":
    main(sys.argv[1:])
