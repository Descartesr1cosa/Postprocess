"""Command-line interface for inspection and validation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .case import MPCNSCase
from .manifest import load_manifest
from .tecplot import export_case_tecplot

def _report(report):
    for x in report.issues: print(f"[{x.severity.upper()}] {x.category}: {x.message}")
    return 0 if report.ok else 1

def main(argv=None) -> int:
    """Run the mpcns-post CLI."""
    p=argparse.ArgumentParser(prog="mpcns-post"); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inspect-manifest","validate-static","validate-restart","validate-case","summary","export-tecplot"):
        q=sub.add_parser(name); q.add_argument("case_directory",type=Path)
        if name in ("validate-restart","validate-case","summary","export-tecplot"): q.add_argument("--data-dir",type=Path)
        if name=="export-tecplot":
            q.add_argument("--output-dir",type=Path,required=True)
            q.add_argument("--prefix",default="mpcns")
            q.add_argument("--location",choices=("node","cell"),default="node")
            q.add_argument("--illuminated-frequency",type=float,default=5.0e-5)
            q.add_argument("--shadow-frequency",type=float,default=1.0e-5)
    a=p.parse_args(argv)
    if a.command=="inspect-manifest":
        m=load_manifest(a.case_directory); print(f"format: {m.format_name} v{m.format_version}\ncase UUID: {m.case_uuid}\nmesh UUID: {m.mesh_uuid}\nranks: {m.number_of_ranks}\nblocks: {m.number_of_blocks}\nfields: {', '.join(x['name'] for x in m.fields)}\noperators: {', '.join(x['name'] for x in m.operators)}"); return 0
    case=MPCNSCase.open(a.case_directory)
    if a.command=="validate-static": return _report(case.validate_static())
    if a.command=="validate-restart":
        report=case.validate_all(data_dir=a.data_dir); report.issues=[x for x in report.issues if x.category not in {"topology","reconstruction","static"}]; return _report(report)
    if a.command=="validate-case": return _report(case.validate_all(data_dir=a.data_dir))
    if a.command=="export-tecplot":
        result=export_case_tecplot(case,data_dir=a.data_dir,output_dir=a.output_dir,prefix=a.prefix,location=a.location,illuminated_frequency=a.illuminated_frequency,shadow_frequency=a.shadow_frequency)
        for item in result["files"].values(): print(f"{item['path']}: {item['zones']} zones, {item['bytes']} bytes")
        return 0
    print(case.summary(data_dir=a.data_dir)); return 0

if __name__=="__main__": raise SystemExit(main())
