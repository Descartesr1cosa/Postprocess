"""Tests for Tecplot serialization and cell-to-node projection."""

from types import SimpleNamespace

import numpy as np

from mpcns_post.tecplot import (
    TecplotZone,
    inspect_tecplot_binary,
    project_cell_values_to_nodes,
    write_tecplot_binary,
)
from mpcns_post.types import ScalarReconstructionOperator


def test_tecplot_binary_roundtrip(tmp_path):
    shape = (2, 3, 4)
    x = np.arange(np.prod(shape), dtype=float).reshape(shape, order="F")
    zone = TecplotZone(
        "rank0000_block0000_Fluid",
        "Fluid",
        {"X": x, "Q": 2 * x},
    )

    path = write_tecplot_binary(
        tmp_path / "roundtrip.plt",
        title="test",
        variable_names=("X", "Q"),
        zones=[zone],
        solution_time=1.5,
    )
    info = inspect_tecplot_binary(path)

    assert info.variables == ("X", "Q")
    assert info.zones == (("rank0000_block0000_Fluid", shape),)


def test_topology_csr_node_projection_non_eight_valence():
    # Node 10 touches three Cells, Node 11 only Cell 2. No fixed 8-Cell stencil.
    operator = ScalarReconstructionOperator(
        "cell_scalar_to_node",
        np.array([10, 11]),
        np.array([0, 3, 4]),
        np.array([0, 1, 2, 2]),
        np.array([0.2, 0.3, 0.5, 1.0]),
    )
    case = SimpleNamespace(
        geometry=SimpleNamespace(
            cell_gid=np.array([0, 1, 2]),
            node_gid=np.array([10, 11]),
        ),
        reconstruction=SimpleNamespace(cell_scalar_to_node=operator),
    )
    values = np.array([2.0, 4.0, 8.0])

    np.testing.assert_allclose(
        project_cell_values_to_nodes(case, values),
        [5.6, 8.0],
    )
    # Excluding Cell 2 renormalizes Node 10 over its true remaining neighbors.
    got = project_cell_values_to_nodes(
        case,
        values,
        valid_mask=np.array([True, True, False]),
    )
    np.testing.assert_allclose(got[0], 3.2)
    assert np.isnan(got[1])
