import struct,numpy as np
from mpcns_post.manifest import load_manifest
from mpcns_post.restart import read_rank_restart

def test_cpp_i_j_k_component_order(manifest_file,tmp_path):
    m=load_manifest(manifest_file); p=tmp_path/"r.bin"; ni,nj,nk,nc=2,3,4,5
    data=bytearray(b"MPCNSRST"+struct.pack("<iidii",1,7,.5,1,1))
    data+=struct.pack("<i",3)+b"U_H"+struct.pack("<iii",0,nc,0)+struct.pack("<7i",0,0,0,ni,nj,nk,1)
    expected=np.empty((ni,nj,nk,nc))
    vals=[]
    for i in range(ni):
      for j in range(nj):
       for k in range(nk):
        for q in range(nc): vals.append(10**6*q+10**4*k+10**2*j+i); expected[i,j,k,q]=vals[-1]
    data+=np.asarray(vals,dtype="<f8").tobytes(); p.write_bytes(data)
    got=read_rank_restart(p,rank=0,manifest=m)
    np.testing.assert_array_equal(got.fields["U_H"].blocks[0].values,expected)

