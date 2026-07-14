from pathlib import Path
import numpy as np
from mpcns_post import MPCNSCase

case=MPCNSCase.open(Path("/path/to/DATA_bin"))
restart=case.read_latest_restart(data_dir=Path("/path/to/DATA"))
fields=case.assemble_dynamic_fields(restart)
B_cell=case.reconstruct_B_cell(fields)
print("B_cell shape:",B_cell.shape)
print("B magnitude max:",np.linalg.norm(B_cell,axis=1).max())

