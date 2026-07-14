"""Reader for the generic MPCNSBIN static section container."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .binary import BinaryReader
from .errors import BinaryFormatError
from .types import BinarySection, Manifest

FILE_TYPES = {"geometry":1,"topology":2,"reconstruction":3,"constant_field":4}
SCALAR_TYPES = {1:np.dtype("i1"),2:np.dtype("u1"),3:np.dtype("i4"),4:np.dtype("u4"),5:np.dtype("i8"),6:np.dtype("u8"),7:np.dtype("f8")}

@dataclass(frozen=True)
class StaticFile:
    file_type: str
    version: int
    case_uuid: str
    mesh_uuid: str
    sections: dict[str, BinarySection]


def _uuid(hi: int, lo: int) -> str: return f"{hi:016x}{lo:016x}"


def read_static_file(path: str | Path, *, expected_type: str, manifest: Manifest) -> StaticFile:
    """Read and validate one generic static rank file."""
    if expected_type not in FILE_TYPES: raise ValueError(f"unknown static file type {expected_type!r}")
    r=BinaryReader(path,endianness=manifest.endianness)
    if r.read_ascii(8) != "MPCNSBIN": raise BinaryFormatError(f"{r.path} at offset 0: bad static magic")
    version=r.read_uint32(); file_type=r.read_uint32(); header_bytes=r.read_uint64(); payload_bytes=r.read_uint64()
    cu=_uuid(r.read_uint64(),r.read_uint64()); mu=_uuid(r.read_uint64(),r.read_uint64())
    endian=r.read_uint32(); float_bytes=r.read_uint32(); index_bytes=r.read_uint32(); reserved=r.read_uint32()
    if version != manifest.format_version or file_type != FILE_TYPES[expected_type]: raise BinaryFormatError(f"{r.path}: static type/version mismatch")
    if header_bytes != 80 or payload_bytes != r.remaining_bytes(): raise BinaryFormatError(f"{r.path}: header/payload byte count mismatch")
    if cu != manifest.case_uuid or mu != manifest.mesh_uuid: raise BinaryFormatError(f"{r.path}: UUID mismatch")
    if (endian,float_bytes,index_bytes,reserved)!=(1,8,8,0): raise BinaryFormatError(f"{r.path}: unsupported scalar/header flags")
    sections={}
    while r.remaining_bytes():
        start=r.tell()
        raw=r.read_bytes(32)
        if b"\0" in raw: name_raw, tail=raw.split(b"\0",1)
        else: name_raw, tail=raw,b""
        if tail and any(tail): raise BinaryFormatError(f"{r.path} at offset {start}: nonzero section-name padding")
        try: name=name_raw.decode("ascii")
        except UnicodeDecodeError as exc: raise BinaryFormatError(f"{r.path} at offset {start}: invalid section name") from exc
        code=r.read_uint32(); components=r.read_uint32(); count=r.read_uint64(); nbytes=r.read_uint64()
        if not name: raise BinaryFormatError(f"{r.path} at offset {start}: empty section name")
        # The current topology writer reuses edge_global_id/face_global_id for
        # the reverse-incidence row keys. Preserve both under an unambiguous
        # compatibility name; every other duplicate remains a hard error.
        if name in sections and expected_type == "topology" and name in {"edge_global_id","face_global_id"}:
            name=name.removesuffix("_global_id")+"_cell_global_id"
        if name in sections: raise BinaryFormatError(f"{r.path} at offset {start}: duplicate section {name!r}")
        if code not in SCALAR_TYPES or components <= 0: raise BinaryFormatError(f"{r.path} at offset {start}: invalid section type/components")
        dt=SCALAR_TYPES[code]
        expected=count*components*dt.itemsize
        if nbytes != expected: raise BinaryFormatError(f"{r.path} at offset {start}: section {name} bytes {nbytes} != {expected}")
        values=r.read_array(dt,count*components)
        values=values.reshape((count,components)) if components>1 else values.reshape(count)
        sections[name]=BinarySection(name,dt,components,count,values)
    r.expect_eof()
    return StaticFile(expected_type,version,cu,mu,sections)
