"""Module execution entrypoint for ``python -m swarm``."""

from __future__ import annotations

import sys

from .cli import main


def _run() -> None:
    sys.exit(main())


if __name__ == "__main__":
    _run()
