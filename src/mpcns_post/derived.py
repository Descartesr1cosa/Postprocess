"""Physics-derived variables and normalization unit conversion."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from .errors import ValidationError

@dataclass
class PrimitiveSpecies:
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray | None = None


def conserved_to_primitive(conserved, *, gamma: float, density_floor: float | None=None) -> PrimitiveSpecies:
    """Convert species [rho,rho*u,E] to primitive state without magnetic energy."""
    q=np.asarray(conserved,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=5 or not np.all(np.isfinite(q)): raise ValidationError("conserved state must be finite with shape (N,5)")
    rho=q[:,0].copy()
    if density_floor is None:
        if np.any(rho<=0): raise ValidationError(f"non-positive density in {np.count_nonzero(rho<=0)} cells")
        denom=rho
    else:
        if density_floor<=0: raise ValueError("density_floor must be positive")
        denom=np.maximum(rho,density_floor); rho=denom.copy()
    velocity=q[:,1:4]/denom[:,None]
    pressure=(gamma-1.0)*(q[:,4]-0.5*denom*np.einsum("ij,ij->i",velocity,velocity))
    if not np.all(np.isfinite(pressure)): raise ValidationError("non-finite pressure")
    return PrimitiveSpecies(rho,velocity,pressure)


class UnitConverter:
    """Convert nondimensional solver arrays with manifest reference values."""
    _refs={"length":"length_ref","time":"time_ref","velocity":"velocity_ref","density":"density_ref","pressure":"pressure_ref","magnetic_field":"magnetic_field_ref","electric_field":"electric_field_ref","current_density":"current_density_ref"}
    _units={"length":{"m":1,"km":1e-3,"Mercury radius":1/2_440_000,"R_M":1/2_440_000},"time":{"s":1},"velocity":{"m/s":1,"km/s":1e-3},"density":{"kg/m^3":1},"pressure":{"Pa":1,"nPa":1e9},"magnetic_field":{"T":1,"nT":1e9},"electric_field":{"V/m":1,"mV/m":1e3},"current_density":{"A/m^2":1,"nA/m^2":1e9}}
    def __init__(self, normalization: Mapping[str,float]): self.normalization=normalization
    def convert(self, values, quantity: str, unit: str):
        """Return SI-scaled values converted to a requested unit."""
        if quantity=="number_density": raise ValueError("use number_density(values, particle_mass, unit)")
        try: scale=self.normalization[self._refs[quantity]]*self._units[quantity][unit]
        except KeyError as exc: raise ValueError(f"unsupported quantity/unit {quantity}/{unit}") from exc
        return np.asarray(values)*scale
    def number_density(self, density, *, particle_mass: float, unit: str="m^-3"):
        """Convert nondimensional mass density to number density."""
        factor={"m^-3":1.0,"cm^-3":1e-6}.get(unit)
        if factor is None or particle_mass<=0: raise ValueError("invalid number-density unit or particle mass")
        return np.asarray(density)*self.normalization["density_ref"]/particle_mass*factor


def species_temperature_kelvin(
    density: np.ndarray,
    pressure: np.ndarray,
    *,
    density_ref: float,
    pressure_ref: float,
    particle_mass: float,
) -> np.ndarray:
    """Return ideal-gas species temperature from nondimensional rho and p.

    MPCNS evolves one monatomic pressure per ion species, so
    ``T = p_phys * m_particle / (rho_phys * k_B)``.
    """
    rho=np.asarray(density,dtype=np.float64); p=np.asarray(pressure,dtype=np.float64)
    if rho.shape != p.shape or density_ref <= 0 or pressure_ref <= 0 or particle_mass <= 0:
        raise ValueError("invalid species-temperature inputs")
    k_b=1.380649e-23
    out=np.full(rho.shape,np.nan,dtype=np.float64); valid=(rho>0)&np.isfinite(rho)&np.isfinite(p)
    out[valid]=(p[valid]*pressure_ref)*particle_mass/(rho[valid]*density_ref*k_b)
    return out


def neutral_sodium_from_photo_rate(
    production_rate_cm3_s: np.ndarray,
    cell_xyz_rm: np.ndarray,
    *,
    illuminated_frequency: float=5.0e-5,
    shadow_frequency: float=1.0e-5,
) -> np.ndarray:
    """Recover neutral Na concentration [cm^-3] from saved production rate.

    This mirrors Mercury's source definition ``q = nu * n_Na``. The geometric
    shadow is ``x < 0`` and ``sqrt(y**2+z**2) < 1`` in Mercury-radius units.
    """
    q=np.asarray(production_rate_cm3_s,dtype=np.float64); xyz=np.asarray(cell_xyz_rm,dtype=np.float64)
    if xyz.shape != (q.size,3) or illuminated_frequency<=0 or shadow_frequency<=0:
        raise ValueError("invalid neutral-sodium inputs")
    shadow=(xyz[:,0]<0)&(np.hypot(xyz[:,1],xyz[:,2])<1.0)
    nu=np.where(shadow,shadow_frequency,illuminated_frequency)
    out=q/nu
    if np.any(~np.isfinite(out)) or np.any(out<0): raise ValidationError("invalid recovered neutral sodium concentration")
    return out


def curvilinear_curl(cell_xyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute Cartesian curl on a structured curvilinear cell-center grid.

    Both arrays have shape ``(ni,nj,nk,3)``. Coordinates and vector values may
    be nondimensional; the returned curl then uses their corresponding ratio.
    Singular coordinate Jacobians use a local Moore-Penrose inverse.
    """
    xyz=np.asarray(cell_xyz,dtype=np.float64); vec=np.asarray(vector,dtype=np.float64)
    if xyz.shape != vec.shape or xyz.ndim!=4 or xyz.shape[-1]!=3 or min(xyz.shape[:3])<2:
        raise ValidationError("curvilinear curl requires matching (ni,nj,nk,3) arrays with axes >= 2")
    edge_order=2 if min(xyz.shape[:3])>=3 else 1
    # M[..., computational_axis, Cartesian_component] = d x_component / d axis.
    m=np.stack([np.stack(np.gradient(xyz[...,c],axis=(0,1,2),edge_order=edge_order),axis=-1) for c in range(3)],axis=-1)
    # Reorder from [..., Cartesian_component, computational_axis].
    m=np.swapaxes(m,-1,-2)
    gradients=np.empty(xyz.shape[:3]+(3,3),dtype=np.float64)
    det=np.linalg.det(m); scale=np.linalg.norm(m,axis=(-2,-1)); good=np.abs(det)>1e-12*np.maximum(scale**3,1e-300)
    for component in range(3):
        dcomp=np.stack(np.gradient(vec[...,component],axis=(0,1,2),edge_order=edge_order),axis=-1)
        gradients[good,component,:]=np.linalg.solve(m[good],dcomp[good,...,None])[...,0]
        if np.any(~good): gradients[~good,component,:]=np.einsum("...ij,...j->...i",np.linalg.pinv(m[~good]),dcomp[~good])
    curl=np.empty_like(vec)
    curl[...,0]=gradients[...,2,1]-gradients[...,1,2]
    curl[...,1]=gradients[...,0,2]-gradients[...,2,0]
    curl[...,2]=gradients[...,1,0]-gradients[...,0,1]
    if np.any(~np.isfinite(curl)): raise ValidationError("curvilinear curl produced non-finite values")
    return curl
