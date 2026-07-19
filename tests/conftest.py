import json

import pytest


@pytest.fixture
def manifest_dict():
    return {
        "format_name": "MPCNS_PostData",
        "format_version": 1,
        "case_uuid": "0" * 32,
        "mesh_uuid": "1" * 32,
        "endianness": "little",
        "float_type": "float64",
        "index_type": "int64",
        "dimension": 3,
        "number_of_blocks": 1,
        "number_of_ranks": 1,
        "array_order": "C",
        "logical_index_order": "i-fastest",
        "linear_index": "i + ni * (j + nj * k)",
        "face_magnetic_semantics": "oriented_face_2form_flux",
        "normalization": {"density_ref": 1.0},
        "physical_constants": {"gamma": 5 / 3},
        "species": ["H"],
        "files": {
            "geometry": ["geometry_0000.bin"],
            "topology": ["topology_0000.bin"],
            "reconstruction": ["reconstruction_0000.bin"],
            "constant_field": ["constant_field_0000.bin"],
        },
        "existing_dynamic_data": {
            "magic": "MPCNSRST",
            "format_version": 1,
            "path_pattern": "./DATA/flow_field{rank:04d}.bin",
            "number_of_rank_files": 1,
            "inactive_semantics": "not applicable outside physics domain",
            "fields": [
                {
                    "name": "U_H",
                    "location": "cell",
                    "location_code": 0,
                    "components": 5,
                    "nghost": 0,
                    "physics_domain": "Fluid",
                }
            ],
        },
        "block_physics_codes": {"1": "Fluid", "2": "Solid"},
        "cell_flag_bits": {"fluid": 1, "solid": 2},
        "operators": [{"name": "B_face_to_cell_cartesian"}],
        "fields": [
            {
                "name": "Photo_rate",
                "location": "cell",
                "components": 1,
                "section_prefix": "field_0000",
            }
        ],
    }


@pytest.fixture
def manifest_file(tmp_path, manifest_dict):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_dict))
    return path
