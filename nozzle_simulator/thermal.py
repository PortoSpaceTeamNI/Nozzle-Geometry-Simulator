"""Adiabatic-wall recovery temperature and diagnostic Bartz coefficient."""

import math

import numpy as np

from .boundary_layer import adiabatic_wall_temperature
from .models import CEAProperties, FlowResult, GeometryResult, NozzleInputs, ThermalResult


def compute_thermal(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
) -> ThermalResult:
    rt = inputs.throat_radius_m
    diameter = 2.0 * rt
    throat_area = math.pi * rt**2
    local_area = math.pi * geometry.radius_m**2
    gamma = flow.gamma
    mach = flow.mach
    chamber_temperature = cea.chamber.temperature_k
    transport = cea.throat
    taw = adiabatic_wall_temperature(geometry, flow, cea)
    wall_temperature = taw.copy()
    sigma = (
        0.5 * (wall_temperature / chamber_temperature)
        * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
        + 0.5
    ) ** -0.68 * (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** -0.12

    constant = (
        0.026 / diameter**0.2
        * (transport.viscosity_pa_s**0.2 * transport.cp_j_kg_k / transport.prandtl**0.6)
        * ((inputs.chamber_pressure_bar * 1e5 / cea.cstar_m_s) ** 0.8)
        * ((diameter / (0.4 * rt)) ** 0.1)
    )
    hg = constant * (throat_area / local_area) ** 0.9 * sigma
    # Adiabatic gas-side condition: Tw = Tr = Taw and therefore qw = 0.
    # hg remains a useful diagnostic for a future conjugate cooling model.
    heat_flux = np.zeros_like(taw)
    return ThermalResult(
        adiabatic_wall_temperature_k=taw,
        wall_temperature_k=wall_temperature,
        heat_transfer_coefficient_w_m2_k=hg,
        heat_flux_w_m2=heat_flux,
    )
