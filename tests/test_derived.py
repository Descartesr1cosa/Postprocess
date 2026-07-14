import numpy as np
from mpcns_post.derived import conserved_to_primitive

def test_species_energy_excludes_magnetic_energy():
    rho=2.; u=np.array([1.,2.,3.]); p=5.; gamma=5/3
    E=.5*rho*np.dot(u,u)+p/(gamma-1); q=np.r_[rho,rho*u,E][None]
    got=conserved_to_primitive(q,gamma=gamma)
    np.testing.assert_allclose(got.velocity,u[None]); np.testing.assert_allclose(got.pressure,p)

