import numpy as np
from types import SimpleNamespace

from mpcns_post.derived import curvilinear_curl,neutral_sodium_from_photo_rate,species_temperature_kelvin
from mpcns_post.tecplot import TecplotZone,inspect_tecplot_binary,project_cell_values_to_nodes,write_tecplot_binary
from mpcns_post.types import ScalarReconstructionOperator


def test_tecplot_binary_roundtrip(tmp_path):
    shape=(2,3,4); x=np.arange(np.prod(shape),dtype=float).reshape(shape,order="F")
    path=write_tecplot_binary(tmp_path/"x.plt",title="test",variable_names=("X","Q"),zones=[TecplotZone("rank0000_block0000_Fluid","Fluid",{"X":x,"Q":2*x})],solution_time=1.5)
    info=inspect_tecplot_binary(path)
    assert info.variables==("X","Q")
    assert info.zones==(("rank0000_block0000_Fluid",shape),)


def test_cartesian_curl():
    a=np.linspace(-1,1,5); x,y,z=np.meshgrid(a,a,a,indexing="ij"); xyz=np.stack((x,y,z),axis=-1)
    # B=(-y/2, x/2, 0) has curl=(0,0,1).
    b=np.stack((-0.5*y,0.5*x,np.zeros_like(z)),axis=-1)
    curl=curvilinear_curl(xyz,b)
    np.testing.assert_allclose(curl,np.broadcast_to([0,0,1],curl.shape),atol=1e-12)


def test_neutral_sodium_and_temperature():
    xyz=np.array([[2.,0,0],[-2.,0,0],[-2.,2,0]])
    neutral=np.array([10.,20.,30.]); nu=np.array([5e-5,1e-5,5e-5])
    np.testing.assert_allclose(neutral_sodium_from_photo_rate(neutral*nu,xyz),neutral)
    # p=n kT and rho=n*m gives exactly T.
    k=1.380649e-23; m=2.; temperature=400.; rho=np.array([3.])*m; p=np.array([3.])*k*temperature
    np.testing.assert_allclose(species_temperature_kelvin(rho,p,density_ref=1,pressure_ref=1,particle_mass=m),temperature)


def test_topology_csr_node_projection_non_eight_valence():
    # Node 10 touches three Cells, Node 11 only Cell 2. No fixed 8-Cell stencil.
    operator=ScalarReconstructionOperator("cell_scalar_to_node",np.array([10,11]),np.array([0,3,4]),np.array([0,1,2,2]),np.array([.2,.3,.5,1.]))
    case=SimpleNamespace(geometry=SimpleNamespace(cell_gid=np.array([0,1,2]),node_gid=np.array([10,11])),reconstruction=SimpleNamespace(cell_scalar_to_node=operator))
    values=np.array([2.,4.,8.])
    np.testing.assert_allclose(project_cell_values_to_nodes(case,values),[5.6,8.])
    # Excluding Cell 2 renormalizes Node 10 over its true remaining neighbors.
    got=project_cell_values_to_nodes(case,values,valid_mask=np.array([True,True,False]))
    np.testing.assert_allclose(got[0],3.2)
    assert np.isnan(got[1])
