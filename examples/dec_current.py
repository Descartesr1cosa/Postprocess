"""Reconstruct DEC current and optionally validate against debug J-edge fields.

Example
-------
python examples/dec_current.py /path/to/DATA_bin --data-dir /path/to/DATA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpcns_post import MPCNSCase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_dir", type=Path, help="expanded DATA_bin directory")
    parser.add_argument("--data-dir", type=Path, help="restart DATA directory")
    parser.add_argument(
        "--validate-debug",
        action="store_true",
        help="require and compare optional J_xi/J_eta/J_zeta debug fields",
    )
    args = parser.parse_args(argv)

    case = MPCNSCase.load(args.static_dir, data_dir=args.data_dir)
    current = case.reconstruct_current_dec(validate_debug=args.validate_debug)
    cell_nA_m2 = case.unit_converter.convert(
        current.cell_vector, "current_density", "nA/m^2"
    )

    print(f"Edge J·dr: {current.edge_1form.shape}")
    print(f"Cell J: {current.cell_vector.shape}")
    print(
        "|J_cell| [nA/m^2]: "
        f"{np.linalg.norm(cell_nA_m2, axis=1).min():.9g} .. "
        f"{np.linalg.norm(cell_nA_m2, axis=1).max():.9g}"
    )
    if current.debug_edge_max_abs_error is not None:
        print(f"debug J_edge max abs error: {current.debug_edge_max_abs_error:.17g}")
    else:
        print("debug J_edge fields: absent (normal production restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
