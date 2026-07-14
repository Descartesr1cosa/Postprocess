import struct,pytest,numpy as np
from mpcns_post.binary import BinaryReader
from mpcns_post.errors import BinaryFormatError

def test_primitives(tmp_path):
    p=tmp_path/"x"; p.write_bytes(struct.pack("<iqd",-3,9,1.25))
    r=BinaryReader(p); assert r.read_int32()==-3; assert r.read_int64()==9; assert r.read_float64()==1.25; r.expect_eof()

def test_truncated_and_tail(tmp_path):
    p=tmp_path/"x"; p.write_bytes(b"abc")
    with pytest.raises(BinaryFormatError): BinaryReader(p).read_int32()
    with pytest.raises(BinaryFormatError): BinaryReader(p).expect_eof()

def test_bad_string_length(tmp_path):
    p=tmp_path/"x"; p.write_bytes(struct.pack("<i",-1))
    with pytest.raises(BinaryFormatError): BinaryReader(p).read_length_prefixed_string()

