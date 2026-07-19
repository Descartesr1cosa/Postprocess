"""Tests for mesh connectivity and reconstruction operators."""

import numpy as np
import pytest

from mpcns_post.errors import ValidationError
from mpcns_post.topology import adjacency_histogram, validate_csr
from mpcns_post.types import CSRConnectivity, VectorReconstructionOperator


class TestTopology:
    def test_valid_three_cell_edge(self):
        offsets = np.array([0, 3])
        indices = np.array([2, 3, 4])

        validate_csr(offsets, indices, name="edge")

        assert adjacency_histogram(CSRConnectivity(offsets, indices)) == {3: 1}

    @pytest.mark.parametrize(
        "offsets",
        [np.array([1, 2]), np.array([0, 2, 1]), np.array([0, 1])],
    )
    def test_bad_offsets(self, offsets):
        with pytest.raises(ValidationError):
            validate_csr(offsets, np.array([0, 1]), name="bad")

    def test_bad_orientation_sign(self):
        with pytest.raises(ValidationError):
            validate_csr(
                np.array([0, 1]),
                np.array([0]),
                signs=np.array([0]),
                name="bad",
            )


class TestReconstruction:
    def test_vector_operator_apply(self):
        operator = VectorReconstructionOperator(
            "identity",
            np.array([0]),
            np.array([0, 3]),
            np.array([0, 1, 2]),
            np.eye(3),
        )

        np.testing.assert_allclose(
            operator.apply(np.array([2.0, 3.0, 4.0])),
            [[2.0, 3.0, 4.0]],
        )
