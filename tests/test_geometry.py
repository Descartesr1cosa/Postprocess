import numpy as np

def test_metric_identity():
    dr=np.array([[3.,4.,0.]])
    np.testing.assert_allclose(np.linalg.norm(dr,axis=1),[5.])

