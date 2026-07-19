"""Inspect and validate an MPCNS Mercury output case in one place.

Examples
--------
Run all checks, including the latest restart files::

    python examples/case_diagnostics.py /path/to/DATA_bin \
        --data-dir /path/to/DATA

Only check the static mesh and reconstruction files::

    python examples/case_diagnostics.py /path/to/DATA_bin --static-only
"""

from __future__ import annotations

import sys

from mpcns_post.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    """Forward to the shared diagnostics command."""
    arguments = sys.argv[1:] if argv is None else argv
    return cli_main(["diagnose", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
