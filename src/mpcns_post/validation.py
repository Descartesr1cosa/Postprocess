"""Structured validation reports and numerical reproduction tests."""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from .assemble import GlobalIDIndex
from .types import GlobalGeometry, GlobalReconstruction

@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    category: str
    message: str
    file: str | None=None
    rank: int | None=None

@dataclass
class ValidationReport:
    issues: list[ValidationIssue]=field(default_factory=list)
    @property
    def ok(self) -> bool: return not any(x.severity=="error" for x in self.issues)
    def add(self,severity: str,category: str,message: str,**context) -> None:
        """Append one issue."""
        self.issues.append(ValidationIssue(severity,category,message,**context))

@dataclass(frozen=True)
class ReconstructionError:
    vector: tuple[float,float,float]
    max_absolute_error: float
    rms_error: float
    ordinary_max_error: float
    axis_touching_max_error: float
    near_axis_shell_max_error: float

@dataclass(frozen=True)
class ReconstructionValidationReport:
    tests: tuple[ReconstructionError,...]
    @property
    def max_absolute_error(self) -> float: return max((x.max_absolute_error for x in self.tests),default=0.0)


def validate_constant_B_reproduction(geometry: GlobalGeometry,reconstruction: GlobalReconstruction,*,test_vectors=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1.2,-.7,.4))) -> ReconstructionValidationReport:
    """Test exact constant Cartesian B reproduction using directed face flux."""
    face_index=GlobalIDIndex.build(geometry.face_gid); cell_index=GlobalIDIndex.build(geometry.cell_gid)
    op=reconstruction.B_face_to_cell; tests=[]
    for vec in test_vectors:
        b0=np.asarray(vec,dtype=np.float64); flux=geometry.face_area_vector@b0
        local=op.apply(flux,face_index); rec=np.full((geometry.cell_gid.size,3),np.nan); rec[cell_index.lookup(op.output_global_ids)]=local
        err=rec-b0; ae=np.abs(err); maxe=float(np.nanmax(ae)); rms=float(np.sqrt(np.nanmean(err*err)))
        # Current v1 cell flags do not distinguish axis shells. Preserve the
        # report schema and mark unavailable regional diagnostics as NaN.
        tests.append(ReconstructionError(tuple(map(float,vec)),maxe,rms,maxe,float("nan"),float("nan")))
    return ReconstructionValidationReport(tuple(tests))


def finite_statistics(values) -> dict[str,float|int]:
    """Compute vectorized finite-value diagnostics."""
    a=np.asarray(values); finite=np.isfinite(a); x=a[finite]
    return {"nan":int(np.isnan(a).sum()),"posinf":int(np.isposinf(a).sum()),"neginf":int(np.isneginf(a).sum()),
      "min":float(x.min()) if x.size else float("nan"),"max":float(x.max()) if x.size else float("nan"),
      "mean":float(x.mean()) if x.size else float("nan"),"rms":float(np.sqrt(np.mean(x*x))) if x.size else float("nan")}

