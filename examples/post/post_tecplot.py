"""Write the standard Fluid Node Tecplot files from data built by post_core."""

import numpy as np

from mpcns_post.errors import ValidationError
from mpcns_post.tecplot import (
    TecplotZone,
    inspect_tecplot_binary,
    write_tecplot_binary,
)

TECPLT_HELPERS = r'''
+def split_vector_fields(fields):
    """
    Expand arrays shaped (N, 3) into x/y/z scalar arrays.

    Put {c} in the vector name where x/y/z should appear.
    """

    output = {}

    for name, values in fields.items():

        # 提前检查 Tecplot 变量名是否为 ASCII
        try:
            name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"Tecplot variable name must be ASCII: {name!r}"
            ) from exc

        array = np.asarray(values)

        if array.ndim == 1:

            if "{c}" in name:
                raise ValueError(
                    f"Scalar field {name!r} must not contain '{{c}}'"
                )

            output[name] = array

        elif array.ndim == 2 and array.shape[1] == 3:

            for component_index, component_name in enumerate(
                ("x", "y", "z")
            ):

                if "{c}" in name:
                    output_name = name.replace(
                        "{c}",
                        component_name,
                    )
                else:
                    output_name = (
                        f"{name}_{component_name}"
                    )

                if output_name in output:
                    raise ValueError(
                        f"Duplicate Tecplot variable name "
                        f"{output_name!r}"
                    )

                output[output_name] = (
                    array[:, component_index]
                )

        else:
            raise ValueError(
                f"{name}: expected shape (N,) or (N,3), "
                f"got {array.shape}"
            )

    return output

def write_fluid_node_tecplot(
    case,
    node_fields,
    output_path,
    title,
):
    """
    Write Node arrays for Fluid blocks only.

    This avoids exporting undefined H+/Na+/electric-field values
    in Solid blocks.
    """

    scalar_fields = split_vector_fields(
        node_fields
    )

    all_fields = {
        "<times><i>x</i> (/R)": case.nodes.coordinates[:, 0],
        "<times><i>y</i> (/R)": case.nodes.coordinates[:, 1],
        "<times><i>z</i> (/R)": case.nodes.coordinates[:, 2],
        **scalar_fields,
    }

    node_blocks = {
        (block.rank, block.block_id): block
        for block in case.iter_blocks(location="node")
    }

    zones = []

    for cell_block in case.iter_blocks(location="cell"):

        if cell_block.physics != "Fluid":
            continue

        key = (
            cell_block.rank,
            cell_block.block_id,
        )

        node_block = node_blocks[key]

        zone_values = {
            name: node_block.reshape(values)
            for name, values in all_fields.items()
        }

        for name, values in zone_values.items():

            if not np.all(np.isfinite(values)):
                bad = np.count_nonzero(
                    ~np.isfinite(values)
                )

                raise ValidationError(
                    f"rank {cell_block.rank}, block {cell_block.block_id}, "
                    f"field {name}: {bad} non-finite Node values"
                )

        zones.append(
            TecplotZone(
                name=(
                    f"rank{cell_block.rank:04d}_"
                    f"block{cell_block.block_id:04d}_Fluid"
                ),
                physics="Fluid",
                values=zone_values,
            )
        )

    if not zones:
        raise RuntimeError(
            "No Fluid block was found"
        )

    solution_time = float(
        case.latest_restart[0].time
    )

    written_path = write_tecplot_binary(
        output_path,
        title=title,
        variable_names=tuple(all_fields),
        zones=zones,
        solution_time=solution_time,
    )

    return inspect_tecplot_binary(
        written_path
    )



'''


def export_node_tecplot(data: dict) -> dict:
    """Write the five standard Node PLT files and return their file info."""
    scope = dict(data)
    # The writer below is defined through exec(), so give its global namespace
    # every dependency explicitly instead of relying on post_core.py imports.
    scope.update(
        np=np,
        TecplotZone=TecplotZone,
        ValidationError=ValidationError,
        inspect_tecplot_binary=inspect_tecplot_binary,
        write_tecplot_binary=write_tecplot_binary,
    )
    exec(TECPLT_HELPERS.replace("\n+", "\n"), scope)
    case = scope["case"]
    output_dir = scope["OUTPUT_DIR"]
    write = scope["write_fluid_node_tecplot"]
    jobs = (
        ("00_fluid_H_Na", "00_fluid_H_Na_node.plt", "NODE_FLUID",
         "MPCNS H+ and Na+ primitive variables at Nodes"),
        ("01_total_B_J_E", "01_em_total_B_J_E_node.plt", "NODE_EM_TOTAL",
         "MPCNS total B, J and total E at Nodes"),
        ("02_B_induced_add", "02_em_B_induced_add_node.plt", "NODE_EM_B_DECOMPOSITION",
         "MPCNS induced and additive magnetic fields at Nodes"),
        ("03_E_components", "03_em_E_components_node.plt", "NODE_EM_E_COMPONENTS",
         "MPCNS electric-field components at Nodes"),
        ("04_auxiliary", "04_em_auxiliary_node.plt", "NODE_EM_AUXILIARY",
         "MPCNS auxiliary electromagnetic quantities at Nodes"),
    )
    infos = {}
    for name, filename, field_name, title in jobs:
        infos[name] = write(case, scope[field_name], output_dir / filename, title)
        print("Written:", infos[name].path)
    return infos
