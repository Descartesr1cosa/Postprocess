"""Manifest-described constant field reader."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .errors import ValidationError
from .static_sections import read_static_file
from .types import Manifest, RankConstantFields


def read_rank_constant_fields(path: str | Path, *, rank: int, manifest: Manifest) -> RankConstantFields:
    """Read owner-oriented constant fields from one rank chunk."""
    s=read_static_file(path,expected_type="constant_field",manifest=manifest).sections
    expected=set(); fields={}; gids={}
    for desc in manifest.fields:
        p=str(desc["section_prefix"]); expected.update({p+"_meta",p+"_ids",p+"_values"})
        if not {p+"_meta",p+"_ids",p+"_values"} <= s.keys(): raise ValidationError(f"{path}: missing sections for field {desc['name']}")
        meta=s[p+"_meta"].values.reshape(-1); ids=s[p+"_ids"].values; vals=s[p+"_values"].values
        comps=int(desc["components"])
        if meta.size!=3 or int(meta[1])!=comps or vals.shape != ((ids.size,comps) if comps>1 else (ids.size,)): raise ValidationError(f"{path}: metadata/shape mismatch for {desc['name']}")
        if not np.all(np.isfinite(vals)): raise ValidationError(f"{path}: non-finite constant field {desc['name']}")
        fields[str(desc["name"])]=vals; gids[str(desc["name"])]=ids
    if set(s)!=expected: raise ValidationError(f"{path}: unexpected constant-field sections {sorted(set(s)-expected)}")
    return RankConstantFields(rank,fields,gids)

