import json,pytest
from mpcns_post.errors import ManifestError
from mpcns_post.manifest import load_manifest

def test_valid(manifest_file):
    manifest=load_manifest(manifest_file)
    assert manifest.number_of_ranks==1
    assert manifest.cell_flag_bits=={"fluid":1,"solid":2}
    assert manifest.block_physics_codes["2"]=="Solid"

@pytest.mark.parametrize(("key","value"),[("format_name","bad"),("case_uuid","xyz"),("face_magnetic_semantics","vector")])
def test_invalid(tmp_path,manifest_dict,key,value):
    manifest_dict[key]=value; p=tmp_path/"m.json"; p.write_text(json.dumps(manifest_dict))
    with pytest.raises(ManifestError): load_manifest(p)

def test_duplicate_field(tmp_path,manifest_dict):
    manifest_dict["fields"]*=2; p=tmp_path/"m.json"; p.write_text(json.dumps(manifest_dict))
    with pytest.raises(ManifestError): load_manifest(p)
