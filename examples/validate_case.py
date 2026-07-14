from pathlib import Path
from mpcns_post import MPCNSCase

case=MPCNSCase.open(Path("/path/to/DATA_bin"))
report=case.validate_all(data_dir=Path("/path/to/DATA"))
for issue in report.issues:
    print(f"[{issue.severity.upper()}] {issue.category}: {issue.message}")
if not report.ok:
    raise SystemExit(1)
print(case.summary(data_dir=Path("/path/to/DATA")))

