"""Export MPCNS Mercury output to Tecplot 112 binary files.

Example::

    python examples/export_tecplot.py /path/to/DATA_bin \
        --data-dir /path/to/DATA \
        --output-dir /path/to/post_output \
        --prefix mercury
"""

from __future__ import annotations

import sys

from mpcns_post.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    """Forward to the shared Tecplot export command."""
    arguments = sys.argv[1:] if argv is None else argv
    return cli_main(["export-tecplot", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
