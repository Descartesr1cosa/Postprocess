"""Command-line interface for diagnostics and data export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .case import MPCNSCase
from .manifest import load_manifest
from .tecplot import export_case_tecplot, export_fields_tecplot


def _print_report(report) -> int:
    for issue in report.issues:
        print(f"[{issue.severity.upper()}] {issue.category}: {issue.message}")
    return 0 if report.ok else 1


def _add_case_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "case_directory",
        type=Path,
        help="directory containing manifest.json",
    )


def _add_data_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory containing flow_field####.bin (auto-detected when omitted)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the shared parser used by the console and example scripts."""
    parser = argparse.ArgumentParser(prog="mpcns-post")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-manifest",
        help="show manifest metadata without opening rank files",
    )
    _add_case_directory(inspect_parser)

    static_parser = subparsers.add_parser(
        "validate-static",
        help="validate static mesh and reconstruction files",
    )
    _add_case_directory(static_parser)

    restart_parser = subparsers.add_parser(
        "validate-restart",
        help="validate the latest restart and assembled fields",
    )
    _add_case_directory(restart_parser)
    _add_data_directory(restart_parser)

    case_parser = subparsers.add_parser(
        "validate-case",
        help="run all static and restart checks",
    )
    _add_case_directory(case_parser)
    _add_data_directory(case_parser)

    summary_parser = subparsers.add_parser(
        "summary",
        help="summarize static data and the latest restart",
    )
    _add_case_directory(summary_parser)
    _add_data_directory(summary_parser)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="print a case summary and all requested checks together",
    )
    _add_case_directory(diagnose_parser)
    _add_data_directory(diagnose_parser)
    diagnose_parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip restart, field assembly, and primitive-variable checks",
    )

    export_parser = subparsers.add_parser(
        "export-tecplot",
        help="write fluid and electromagnetic Tecplot 112 binary files",
    )
    _add_case_directory(export_parser)
    _add_data_directory(export_parser)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--prefix", default="mpcns")
    export_parser.add_argument(
        "--location",
        choices=("node", "cell"),
        default="node",
    )
    export_parser.add_argument("--illuminated-frequency", type=float, default=5.0e-5)
    export_parser.add_argument("--shadow-frequency", type=float, default=1.0e-5)

    fields_parser = subparsers.add_parser(
        "export-fields",
        help="export named raw/derived fields through the general Tecplot API",
    )
    _add_case_directory(fields_parser)
    _add_data_directory(fields_parser)
    fields_parser.add_argument("fields", nargs="+", help="public case field names")
    fields_parser.add_argument("--output", type=Path, required=True)
    fields_parser.add_argument(
        "--location",
        choices=("cell", "node", "face", "edge"),
        default="cell",
    )
    return parser


def _inspect_manifest(case_directory: Path) -> int:
    manifest = load_manifest(case_directory)
    print(f"format: {manifest.format_name} v{manifest.format_version}")
    print(f"case UUID: {manifest.case_uuid}")
    print(f"mesh UUID: {manifest.mesh_uuid}")
    print(f"ranks: {manifest.number_of_ranks}")
    print(f"blocks: {manifest.number_of_blocks}")
    print(f"fields: {', '.join(field['name'] for field in manifest.fields)}")
    print(f"operators: {', '.join(op['name'] for op in manifest.operators)}")
    return 0


def _export_tecplot(case: MPCNSCase, args: argparse.Namespace) -> int:
    result = export_case_tecplot(
        case,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        location=args.location,
        illuminated_frequency=args.illuminated_frequency,
        shadow_frequency=args.shadow_frequency,
    )
    for label, info in result["files"].items():
        print(
            f"{label}: {info['path']} "
            f"({info['zones']} zones, {info['bytes']} bytes)"
        )
    summary = args.output_dir / f"{args.prefix}_export_summary_{args.location}.json"
    print(f"summary: {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the mpcns-post CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "inspect-manifest":
        return _inspect_manifest(args.case_directory)

    case = MPCNSCase.open(args.case_directory)
    if args.command == "validate-static":
        return _print_report(case.validate_static())
    if args.command == "validate-restart":
        report = case.validate_all(data_dir=args.data_dir)
        report.issues = [
            issue
            for issue in report.issues
            if issue.category not in {"topology", "reconstruction", "static"}
        ]
        return _print_report(report)
    if args.command == "validate-case":
        return _print_report(case.validate_all(data_dir=args.data_dir))
    if args.command == "summary":
        print(case.summary(data_dir=args.data_dir))
        return 0
    if args.command == "diagnose":
        report = (
            case.validate_static()
            if args.static_only
            else case.validate_all(data_dir=args.data_dir)
        )
        print(
            case.summary(
                data_dir=args.data_dir,
                include_restart=not args.static_only,
            )
        )
        print("\nDiagnostics:")
        status = _print_report(report)
        if status == 0:
            print("[OK] all requested checks passed")
        return status
    if args.command == "export-tecplot":
        return _export_tecplot(case, args)
    if args.command == "export-fields":
        case.load_latest(data_dir=args.data_dir)
        fields = {name: case.get_field(name) for name in args.fields}
        info = export_fields_tecplot(
            case,
            fields,
            args.output,
            location=args.location,
        )
        print(f"{info.path}: {len(info.zones)} zones, {len(info.variables)} variables")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
