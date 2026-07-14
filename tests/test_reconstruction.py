import numpy as np
from mpcns_post.types import VectorReconstructionOperator

def test_vector_apply():
    op=VectorReconstructionOperator("x",np.array([0]),np.array([0,3]),np.array([0,1,2]),np.eye(3))
    np.testing.assert_allclose(op.apply(np.array([2.,3.,4.])),[[2,3,4]])

