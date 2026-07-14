from pathlib import Path
from mpcns_post import MPCNSCase

case=MPCNSCase.open(Path("/path/to/DATA_bin"))
print(case.summary())

