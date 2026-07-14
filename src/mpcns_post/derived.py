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

