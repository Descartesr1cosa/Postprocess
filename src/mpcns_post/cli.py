"""Command-line interface for inspection and validation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .case import MPCNSCase
from .manifest import load_manifest

def _report(report):
    for x in report.issues: print(f"[{x.severity.upper()}] {x.category}: {x.message}")
    return 0 if report.ok else 1

def main(argv=None) -> int:
    """Run the mpcns-post CLI."""
    p=argparse.ArgumentParser(prog="mpcns-post"); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inspect-manifest","validate-static","validate-restart","validate-case","summary"):
        q=sub.add_parser(name); q.add_argument("case_directory",type=Path)
        if name in ("validate-restart","validate-case","summary"): q.add_argument("--data-dir",type=Path)
    a=p.parse_args(argv)
    if a.command=="inspect-manifest":
        m=load_manifest(a.case_directory); print(f"format: {m.format_name} v{m.format_version}\ncase UUID: {m.case_uuid}\nmesh UUID: {m.mesh_uuid}\nranks: {m.number_of_ranks}\nblocks: {m.number_of_blocks}\nfields: {', '.join(x['name'] for x in m.fields)}\noperators: {', '.join(x['name'] for x in m.operators)}"); return 0
    case=MPCNSCase.open(a.case_directory)
    if a.command=="validate-static": return _report(case.validate_static())
    if a.command=="validate-restart":
        report=case.validate_all(data_dir=a.data_dir); report.issues=[x for x in report.issues if x.category not in {"topology","reconstruction","static"}]; return _report(report)
    if a.command=="validate-case": return _report(case.validate_all(data_dir=a.data_dir))
    print(case.summary(data_dir=a.data_dir)); return 0

if __name__=="__main__": raise SystemExit(main())

