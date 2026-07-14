"""Manifest loading and strict schema validation."""
from __future__ import annotations
import json, math, re
from pathlib import Path
from .errors import ManifestError
from .types import Manifest

SUPPORTED_FACE_MAGNETIC_SEMANTICS = {"oriented_face_2form_flux"}
STATIC_KEYS = ("geometry", "topology", "reconstruction", "constant_field")


def _require(data: dict, key: str):
    if key not in data: raise ManifestError(f"manifest: missing required key {key!r}")
    return data[key]


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a version-1 MPCNS post-data manifest."""
    p = Path(path)
    if p.is_dir(): p = p / "manifest.json"
    try: data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ManifestError(f"{p}: cannot load manifest: {exc}") from exc
    expected = {"format_name":"MPCNS_PostData", "format_version":1, "endianness":"little",
                "float_type":"float64", "index_type":"int64", "dimension":3}
    for key, value in expected.items():
        if _require(data,key) != value: raise ManifestError(f"{p}: {key} must be {value!r}, got {data[key]!r}")
    uuid_re = re.compile(r"^[0-9a-fA-F]{32}$")
    for key in ("case_uuid", "mesh_uuid"):
        if not isinstance(_require(data,key),str) or not uuid_re.fullmatch(data[key]): raise ManifestError(f"{p}: {key} must be 32 hexadecimal characters")
    ranks = _require(data,"number_of_ranks")
    blocks = _require(data,"number_of_blocks")
    if not isinstance(ranks,int) or isinstance(ranks,bool) or ranks <= 0: raise ManifestError(f"{p}: number_of_ranks must be positive")
    if not isinstance(blocks,int) or isinstance(blocks,bool) or blocks <= 0: raise ManifestError(f"{p}: number_of_blocks must be positive")
    files = _require(data,"files")
    if not isinstance(files,dict): raise ManifestError(f"{p}: files must be an object")
    for key in STATIC_KEYS:
        seq = files.get(key)
        if not isinstance(seq,list) or len(seq) != ranks or not all(isinstance(x,str) and x for x in seq):
            raise ManifestError(f"{p}: files.{key} must contain exactly {ranks} filenames")
    dynamic = _require(data,"existing_dynamic_data")
    if not isinstance(dynamic,dict) or dynamic.get("number_of_rank_files") != ranks: raise ManifestError(f"{p}: dynamic rank-file count must equal {ranks}")
    norm = _require(data,"normalization")
    phys = _require(data,"physical_constants")
    if not isinstance(norm,dict) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v <= 0 for v in norm.values()):
        raise ManifestError(f"{p}: all normalization values must be finite and positive")
    if not isinstance(phys,dict) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in phys.values()):
        raise ManifestError(f"{p}: all physical constants must be finite")
    fields = _require(data,"fields"); operators = _require(data,"operators")
    for label, seq in (("field",fields),("operator",operators)):
        if not isinstance(seq,list) or any(not isinstance(x,dict) or not isinstance(x.get("name"),str) for x in seq): raise ManifestError(f"{p}: invalid {label} definitions")
        names=[x["name"] for x in seq]
        if len(names) != len(set(names)): raise ManifestError(f"{p}: duplicate {label} name")
    semantics = _require(data,"face_magnetic_semantics")
    if semantics not in SUPPORTED_FACE_MAGNETIC_SEMANTICS: raise ManifestError(f"{p}: unsupported face_magnetic_semantics {semantics!r}")
    return Manifest(p,data["format_name"],data["format_version"],data["case_uuid"].lower(),data["mesh_uuid"].lower(),
        data["endianness"],data["float_type"],data["index_type"],data["dimension"],blocks,ranks,
        _require(data,"array_order"),_require(data,"logical_index_order"),_require(data,"linear_index"),semantics,
        {k:float(v) for k,v in norm.items()},{k:float(v) for k,v in phys.items()},tuple(_require(data,"species")),
        {k:tuple(v) for k,v in files.items()},dynamic,tuple(operators),tuple(fields),
        {str(k):str(v) for k,v in data.get("block_physics_codes",{}).items()},
        {str(k):int(v) for k,v in data.get("cell_flag_bits",{}).items()})
