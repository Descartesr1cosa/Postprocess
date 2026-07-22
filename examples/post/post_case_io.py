"""The small, reusable MPCNS input stage used by the post-processing scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mpcns_post import MPCNSCase


@dataclass(frozen=True)
class LoadedCase:
    """Objects needed by the physics and export modules."""

    case: MPCNSCase
    fluid_mask: object


def load_case(case_dir: Path) -> LoadedCase:
    """Read DATA_bin and DATA under one MPCNS output directory.

    ``case_dir`` should be the directory containing ``DATA_bin`` and ``DATA``.
    Keeping this read step here lets other scripts reuse exactly the same case.
    """
    case_dir = Path(case_dir)
    case = MPCNSCase.load(case_dir / "DATA_bin", data_dir=case_dir / "DATA")
    fluid_mask = case.H.valid_mask & case.Na.valid_mask

    print("Read MPCNS case:", case_dir)
    print("  B_face_to_J_edge:", case.reconstruction.B_face_to_J_edge is not None)
    print("  J_edge_to_cell:", case.reconstruction.J_edge_to_cell is not None)
    return LoadedCase(case=case, fluid_mask=fluid_mask)
