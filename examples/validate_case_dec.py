"""Validate DATA_bin/DATA and exercise solver-equivalent DEC current.

Examples
--------
python examples/validate_case_dec.py /path/to/DATA_bin --data-dir /path/to/DATA
python examples/validate_case_dec.py /path/to/DATA_bin --data-dir /path/to/DATA --validate-debug
python examples/validate_case_dec.py /path/to/DATA_bin --static-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpcns_post import MPCNSCase


def print_report(report) -> None:
    """Print one compact line per structured validation issue."""
    for issue in report.issues:
        context = []
        if issue.file is not None:
            context.append(issue.file)
        if issue.rank is not None:
            context.append(f"rank={issue.rank}")
        suffix = f" ({', '.join(context)})" if context else ""
        print(f"[{issue.severity.upper():7s}] {issue.category}: {issue.message}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_dir", type=Path, help="expanded DATA_bin directory")
    parser.add_argument("--data-dir", type=Path, help="restart DATA directory")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="validate only manifest, geometry, topology, and reconstruction",
    )
    parser.add_argument(
        "--validate-debug",
        action="store_true",
        help="require and compare optional J_xi/J_eta/J_zeta debug fields",
    )
    args = parser.parse_args(argv)

    case = MPCNSCase.open(args.static_dir)
    print(case.summary(include_restart=False))

    if args.static_only:
        report = case.validate_static()
        print_report(report)
        return 0 if report.ok else 1

    report = case.validate_all(data_dir=args.data_dir)
    print_report(report)
    if not report.ok:
        return 1

    # validate_all loads and assembles DATA, so this tests the public DEC API
    # on exactly the restart that passed the format/physics checks above.
    current = case.reconstruct_current_dec(validate_debug=args.validate_debug)
    J_nA_m2 = case.unit_converter.convert(
        current.cell_vector, "current_density", "nA/m^2"
    )
    magnitude = np.linalg.norm(J_nA_m2, axis=1)

    print(f"restart fields: {', '.join(case.dynamic_fields.fields)}")
    print(f"Edge J·dr shape: {current.edge_1form.shape}")
    print(f"Cell J shape: {current.cell_vector.shape}")
    print(f"|J_cell| [nA/m^2]: {magnitude.min():.9g} .. {magnitude.max():.9g}")
    if current.debug_edge_max_abs_error is None:
        print("debug J_edge: absent (normal production restart)")
    else:
        print(f"debug J_edge max abs error: {current.debug_edge_max_abs_error:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
