"""Quasi-one-dimensional flow profiles using CEA station properties."""


import numpy as np
from scipy.optimize import brentq

from .models import CEAProperties, FlowResult, GeometryResult, NozzleInputs


def area_ratio_from_mach(mach: float, gamma: float) -> float:
    return (1.0 / mach) * (
        (2.0 / (gamma + 1.0))
        * (1.0 + (gamma - 1.0) * 0.5 * mach**2)
    ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area_ratio(area_ratio: float, gamma: float, supersonic: bool) -> float:
    area_ratio = max(float(area_ratio), 1.0)
    if area_ratio <= 1.0 + 1e-10:
        return 1.0
    residual = lambda mach: area_ratio_from_mach(mach, gamma) - area_ratio
    if supersonic:
        return float(brentq(residual, 1.0 + 1e-8, 30.0))
    return float(brentq(residual, 1e-8, 1.0 - 1e-8))


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value**2 * (3.0 - 2.0 * value)


def compute_flow(
    inputs: NozzleInputs, geometry: GeometryResult, cea: CEAProperties
) -> FlowResult:
    x = geometry.x_m
    radius = geometry.radius_m
    divergent = x >= geometry.throat_x_m
    local_eps = (radius / inputs.throat_radius_m) ** 2
    progress = _smoothstep((local_eps - 1.0) / (inputs.expansion_ratio - 1.0))
    gamma = np.where(
        divergent,
        cea.throat.gamma + (cea.exit.gamma - cea.throat.gamma) * progress,
        cea.chamber.gamma + (cea.throat.gamma - cea.chamber.gamma)
        * np.clip(x / max(geometry.throat_x_m, 1e-12), 0.0, 1.0),
    )
    mach = np.array([
        mach_from_area_ratio(ar, gam, bool(is_div))
        for ar, gam, is_div in zip(local_eps, gamma, divergent)
    ])
    temperature = cea.chamber.temperature_k / (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    )
    pressure = inputs.chamber_pressure_bar * (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    ) ** (-gamma / (gamma - 1.0))
    return FlowResult(mach=mach, temperature_k=temperature, pressure_bar=pressure, gamma=gamma)

