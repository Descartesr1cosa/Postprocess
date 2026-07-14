import numpy as np,pytest
from mpcns_post.errors import ValidationError
from mpcns_post.topology import adjacency_histogram,validate_csr
from mpcns_post.types import CSRConnectivity

def test_valid_three_cell_edge():
    o=np.array([0,3]); i=np.array([2,3,4]); validate_csr(o,i,name="edge")
    assert adjacency_histogram(CSRConnectivity(o,i))=={3:1}

@pytest.mark.parametrize("offsets",[np.array([1,2]),np.array([0,2,1]),np.array([0,1])])
def test_bad_offsets(offsets):
    with pytest.raises(ValidationError): validate_csr(offsets,np.array([0,1]),name="bad")

def test_bad_sign():
    with pytest.raises(ValidationError): validate_csr(np.array([0,1]),np.array([0]),signs=np.array([0]),name="bad")

