"""Reader for existing per-rank MPCNS restart files."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from .binary import BinaryReader
from .errors import BinaryFormatError
from .types import BlockExtent, Manifest, RankRestart, RestartBlockField, RestartField


def read_rank_restart(path: str | Path, *, rank: int, manifest: Manifest) -> RankRestart:
    """Read a restart written by ``IOModule::WriteRestartBinFile``.

    The C++ writer nests blocks under each field and iterates i, then j, then k,
    then component. Its hi bounds are exclusive.
    """
    r=BinaryReader(path,endianness=manifest.endianness)
    dyn=manifest.existing_dynamic_data
    if r.read_ascii(8) != dyn.get("magic","MPCNSRST"): raise BinaryFormatError(f"{r.path} at offset 0: bad restart magic")
    version=r.read_int32(); step=r.read_int32(); time=r.read_float64(); nblock=r.read_int32(); nfield=r.read_int32()
    if version != dyn.get("format_version",1): raise BinaryFormatError(f"{r.path}: unsupported restart version {version}")
    if not math.isfinite(time) or nblock < 0 or nfield < 0: raise BinaryFormatError(f"{r.path}: invalid restart header")
    specs=dyn.get("fields",[])
    optional_names = {"J_xi", "J_eta", "J_zeta"} if dyn.get("optional_dec_jedge") else set()
    required_specs = [x for x in specs if str(x.get("name")) not in optional_names]
    if nfield not in {len(required_specs), len(specs)}:
        raise BinaryFormatError(
            f"{r.path}: nfield {nfield} must contain required fields ({len(required_specs)}) "
            f"or required plus the complete optional DEC triplet ({len(specs)})"
        )
    actual_specs = specs if nfield == len(specs) else required_specs
    location_by_code={int(x["location_code"]):str(x["location"]) for x in actual_specs}
    fields={}
    for fi in range(nfield):
        name=r.read_length_prefixed_string(); loc=r.read_int32(); comps=r.read_int32(); nghost=r.read_int32()
        spec=actual_specs[fi]
        got=(name,loc,comps,nghost); expected=(spec["name"],int(spec["location_code"]),int(spec["components"]),int(spec["nghost"]))
        if got != expected: raise BinaryFormatError(f"{r.path} at field {fi}: metadata {got!r} != manifest {expected!r}")
        blocks=[]
        for bi in range(nblock):
            lo=tuple(r.read_int32() for _ in range(3)); hi=tuple(r.read_int32() for _ in range(3)); active_i=r.read_int32()
            if active_i not in (0,1) or any(h<l for l,h in zip(lo,hi)): raise BinaryFormatError(f"{r.path} at field {name}, block {bi}: invalid extent/active flag")
            extent=BlockExtent(lo,hi,bool(active_i)); shape=extent.shape
            if active_i:
                raw=r.read_array(np.float64,int(np.prod(shape,dtype=np.int64))*comps)
                values=raw.reshape((*shape,comps))
                if not np.all(np.isfinite(values)): raise BinaryFormatError(f"{r.path}: non-finite values in {name} block {bi}")
            else: values=np.full((*shape,comps),np.nan,dtype=np.float64)
            blocks.append(RestartBlockField(extent,values))
        fields[name]=RestartField(name,loc,location_by_code[loc],comps,nghost,blocks)
    r.expect_eof()
    return RankRestart(rank,step,time,version,fields)
