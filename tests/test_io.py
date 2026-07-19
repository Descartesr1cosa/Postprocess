"""Tests for manifest and binary input formats."""

import json
import struct

import numpy as np
import pytest

from mpcns_post.binary import BinaryReader
from mpcns_post.errors import BinaryFormatError, ManifestError
from mpcns_post.manifest import load_manifest
from mpcns_post.restart import read_rank_restart


class TestBinaryReader:
    def test_primitives(self, tmp_path):
        path = tmp_path / "values.bin"
        path.write_bytes(struct.pack("<iqd", -3, 9, 1.25))

        reader = BinaryReader(path)
        assert reader.read_int32() == -3
        assert reader.read_int64() == 9
        assert reader.read_float64() == 1.25
        reader.expect_eof()

    def test_truncated_input_and_trailing_bytes(self, tmp_path):
        path = tmp_path / "truncated.bin"
        path.write_bytes(b"abc")

        with pytest.raises(BinaryFormatError):
            BinaryReader(path).read_int32()
        with pytest.raises(BinaryFormatError):
            BinaryReader(path).expect_eof()

    def test_negative_string_length(self, tmp_path):
        path = tmp_path / "string.bin"
        path.write_bytes(struct.pack("<i", -1))

        with pytest.raises(BinaryFormatError):
            BinaryReader(path).read_length_prefixed_string()


class TestManifest:
    def test_valid_manifest(self, manifest_file):
        manifest = load_manifest(manifest_file)

        assert manifest.number_of_ranks == 1
        assert manifest.cell_flag_bits == {"fluid": 1, "solid": 2}
        assert manifest.block_physics_codes["2"] == "Solid"

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("format_name", "bad"),
            ("case_uuid", "xyz"),
            ("face_magnetic_semantics", "vector"),
        ],
    )
    def test_invalid_manifest_value(self, tmp_path, manifest_dict, key, value):
        manifest_dict[key] = value
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest_dict))

        with pytest.raises(ManifestError):
            load_manifest(path)

    def test_duplicate_field(self, tmp_path, manifest_dict):
        manifest_dict["fields"] *= 2
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest_dict))

        with pytest.raises(ManifestError):
            load_manifest(path)


class TestRestartReader:
    def test_cpp_i_j_k_component_order(self, manifest_file, tmp_path):
        manifest = load_manifest(manifest_file)
        path = tmp_path / "restart.bin"
        ni, nj, nk, ncomponents = 2, 3, 4, 5
        data = bytearray(b"MPCNSRST" + struct.pack("<iidii", 1, 7, 0.5, 1, 1))
        data += struct.pack("<i", 3) + b"U_H"
        data += struct.pack("<iii", 0, ncomponents, 0)
        data += struct.pack("<7i", 0, 0, 0, ni, nj, nk, 1)

        expected = np.empty((ni, nj, nk, ncomponents))
        values = []
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    for component in range(ncomponents):
                        value = 10**6 * component + 10**4 * k + 10**2 * j + i
                        values.append(value)
                        expected[i, j, k, component] = value

        data += np.asarray(values, dtype="<f8").tobytes()
        path.write_bytes(data)

        restart = read_rank_restart(path, rank=0, manifest=manifest)
        np.testing.assert_array_equal(restart.fields["U_H"].blocks[0].values, expected)
