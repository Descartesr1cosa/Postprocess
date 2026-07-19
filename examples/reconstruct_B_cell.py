"""Reconstruct cell-centered Cartesian magnetic field from face fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpcns_post import MPCNSCase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)

    case = MPCNSCase.load(args.case_directory, data_dir=args.data_dir)
    b_cell = case.reconstruct_B_cell(case.dynamic_fields)
    magnitude = np.linalg.norm(b_cell, axis=1)

    print(f"B_cell shape: {b_cell.shape}")
    print(f"|B| range: [{magnitude.min():.6g}, {magnitude.max():.6g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
