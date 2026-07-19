"""Core rank-local and global data containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
UInt32Array = npt.NDArray[np.uint32]


@dataclass(frozen=True)
class BinarySection:
    name: str
    dtype: np.dtype
    components: int
    count: int
    values: np.ndarray


@dataclass(frozen=True)
class Manifest:
    path: Path
    format_name: str
    format_version: int
    case_uuid: str
    mesh_uuid: str
    endianness: str
    float_type: str
    index_type: str
    dimension: int
    number_of_blocks: int
    number_of_ranks: int
    array_order: str
    logical_index_order: str
    linear_index: str
    face_magnetic_semantics: str
    normalization: Mapping[str, float]
    physical_constants: Mapping[str, float]
    species: tuple[str, ...]
    files: Mapping[str, tuple[str, ...]]
    existing_dynamic_data: Mapping[str, object]
    operators: tuple[Mapping[str, object], ...]
    fields: tuple[Mapping[str, object], ...]
    block_physics_codes: Mapping[str, str] = field(default_factory=dict)
    cell_flag_bits: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockExtent:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]
    active: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        # The C++ FieldBlock uses half-open [lo, hi) extents.
        return tuple(h - l for l, h in zip(self.lo, self.hi))  # type: ignore[return-value]


@dataclass
class RestartBlockField:
    extent: BlockExtent
    values: FloatArray


@dataclass
class RestartField:
    name: str
    location_code: int
    location: str
    components: int
    nghost: int
    blocks: list[RestartBlockField]


@dataclass
class RankRestart:
    rank: int
    step: int
    time: float
    version: int
    fields: dict[str, RestartField]


@dataclass
class RankGeometry:
    rank: int
    node_gid: Int64Array; node_xyz: FloatArray
    edge_gid: Int64Array; edge_node_ids: Int64Array; edge_center_xyz: FloatArray
    edge_dr: FloatArray; edge_length: FloatArray; edge_flags: UInt32Array
    face_gid: Int64Array; face_center_xyz: FloatArray; face_area_vector: FloatArray
    face_area: FloatArray; face_flags: UInt32Array
    cell_gid: Int64Array; cell_center_xyz: FloatArray; cell_volume: FloatArray; cell_flags: UInt32Array


@dataclass
class CSRConnectivity:
    offsets: Int64Array
    indices: Int64Array
    signs: Int32Array | None = None
    row_global_ids: Int64Array | None = None

    def row(self, entity_index: int) -> np.ndarray:
        """Return a zero-copy view of one connectivity row."""
        if not 0 <= entity_index < self.offsets.size - 1:
            raise IndexError(entity_index)
        return self.indices[self.offsets[entity_index]:self.offsets[entity_index + 1]]


@dataclass
class LocalEntityMap:
    block_id: int
    location: str
    logical_shape: tuple[int, int, int]
    global_ids: Int64Array
    orientation_sign: Int32Array
    owner_mask: np.ndarray


@dataclass
class RankTopology:
    rank: int
    face_to_edge: CSRConnectivity
    cell_to_face: CSRConnectivity
    node_to_cell: CSRConnectivity
    edge_to_cell: CSRConnectivity
    face_to_cell: CSRConnectivity
    local_maps: list[LocalEntityMap]
    block_connections: Int64Array


@dataclass
class VectorReconstructionOperator:
    name: str
    output_global_ids: Int64Array
    offsets: Int64Array
    input_global_ids: Int64Array
    weights: FloatArray

    def apply(self, input_values: FloatArray, input_gid_to_index: object | None = None,
              output_count: int | None = None) -> FloatArray:
        """Apply the sparse vector reconstruction operator."""
        if input_gid_to_index is None:
            mapped = self.input_global_ids
        elif isinstance(input_gid_to_index, dict):
            mapped = np.fromiter((input_gid_to_index[int(g)] for g in self.input_global_ids),
                                 dtype=np.int64, count=self.input_global_ids.size)
        elif hasattr(input_gid_to_index, "lookup"):
            mapped = input_gid_to_index.lookup(self.input_global_ids)
        else:
            a = np.asarray(input_gid_to_index)
            mapped = a[self.input_global_ids]
        rows = self.offsets.size - 1
        local = np.empty((rows, 3), dtype=np.float64)
        for i in range(rows):
            sl = slice(self.offsets[i], self.offsets[i + 1])
            local[i] = np.sum(self.weights[sl] * input_values[mapped[sl], None], axis=0)
        if output_count is None:
            return local
        out = np.full((output_count, 3), np.nan)
        out[self.output_global_ids] = local
        return out


@dataclass
class ScalarReconstructionOperator:
    name: str
    output_global_ids: Int64Array
    offsets: Int64Array
    input_global_ids: Int64Array
    weights: FloatArray


@dataclass
class RankReconstruction:
    rank: int
    B_face_to_cell: VectorReconstructionOperator
    cell_scalar_to_node: ScalarReconstructionOperator


@dataclass
class RankConstantFields:
    rank: int
    fields: dict[str, np.ndarray]
    global_ids: dict[str, Int64Array]


@dataclass
class GlobalGeometry:
    node_gid: Int64Array; node_xyz: FloatArray
    edge_gid: Int64Array; edge_node_ids: Int64Array; edge_center_xyz: FloatArray
    edge_dr: FloatArray; edge_length: FloatArray; edge_flags: UInt32Array
    face_gid: Int64Array; face_center_xyz: FloatArray; face_area_vector: FloatArray
    face_area: FloatArray; face_flags: UInt32Array
    cell_gid: Int64Array; cell_center_xyz: FloatArray; cell_volume: FloatArray; cell_flags: UInt32Array


@dataclass
class GlobalTopology:
    local_maps: list[LocalEntityMap]
    node_to_cell: CSRConnectivity
    edge_to_cell: CSRConnectivity
    face_to_cell: CSRConnectivity
    cell_to_face: CSRConnectivity | None = None


@dataclass
class GlobalReconstruction:
    B_face_to_cell: VectorReconstructionOperator
    cell_scalar_to_node: ScalarReconstructionOperator


@dataclass
class GlobalFields:
    fields: dict[str, FloatArray]
    global_ids: dict[str, Int64Array] = field(default_factory=dict)
    valid_masks: dict[str, np.ndarray] = field(default_factory=dict)
