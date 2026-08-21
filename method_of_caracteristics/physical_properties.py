"""Perfect-gas properties used by the axisymmetric MOC solver."""

from __future__ import annotations

import math


def mach_angle(M: float) -> float:
    if not math.isfinite(M) or M <= 1.0:
        raise ValueError("The Mach angle is defined here only for M > 1.")
    return math.asin(1.0 / M)


def prandtl_meyer(M: float, gamma: float) -> float:
    if not math.isfinite(M) or M < 1.0:
        raise ValueError("The Prandtl-Meyer function requires M >= 1.")
    root = math.sqrt(max(M * M - 1.0, 0.0))
    return (
        math.sqrt((gamma + 1.0) / (gamma - 1.0))
        * math.atan(math.sqrt((gamma - 1.0) / (gamma + 1.0)) * root)
        - math.atan(root)
    )


def area_ratio_from_mach(M: float, gamma: float) -> float:
    if M <= 0.0:
        raise ValueError("Mach number must be positive.")
    return (1.0 / M) * (
        (2.0 / (gamma + 1.0))
        * (1.0 + 0.5 * (gamma - 1.0) * M * M)
    ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def pressure_from_mach(M: float, stagnation_pressure_pa: float, gamma: float) -> float:
    return stagnation_pressure_pa * (
        1.0 + 0.5 * (gamma - 1.0) * M * M
    ) ** (-gamma / (gamma - 1.0))


def state_from_pressure(
    pressure_pa: float,
    stagnation_pressure_pa: float,
    stagnation_temperature_k: float,
    gamma: float,
    gas_constant_j_kg_k: float,
) -> tuple[float, float, float, float]:
    """Return ``(temperature, density, velocity, Mach)`` on one isentrope."""
    if not 0.0 < pressure_pa < stagnation_pressure_pa:
        raise ValueError("Static pressure must lie between zero and stagnation pressure.")
    temperature = stagnation_temperature_k * (
        pressure_pa / stagnation_pressure_pa
    ) ** ((gamma - 1.0) / gamma)
    cp = gamma * gas_constant_j_kg_k / (gamma - 1.0)
    velocity_squared = 2.0 * cp * (stagnation_temperature_k - temperature)
    if velocity_squared <= 0.0:
        raise ValueError("The isentropic state produced a non-positive velocity.")
    velocity = math.sqrt(velocity_squared)
    speed_of_sound = math.sqrt(gamma * gas_constant_j_kg_k * temperature)
    M = velocity / speed_of_sound
    density = pressure_pa / (gas_constant_j_kg_k * temperature)
    return temperature, density, velocity, M


def compatibility_Q(M: float, density: float, velocity: float) -> float:
    """Pressure coefficient Q in ``Q dp +/- dtheta + S dx = 0``."""
    if M <= 1.0 or density <= 0.0 or velocity <= 0.0:
        raise ValueError("Compatibility coefficient Q requires a supersonic physical state.")
    return math.sqrt(M * M - 1.0) / (density * velocity * velocity)


def compatibility_S(M: float, theta: float, radius: float, family: str) -> float:
    """Axisymmetric source S+ or S- away from the symmetry axis."""
    if radius <= 0.0:
        raise ValueError("S+/- cannot be evaluated directly at r = 0.")
    mu = mach_angle(M)
    if family == "plus":
        angle = theta + mu
    elif family == "minus":
        angle = theta - mu
    else:
        raise ValueError("Characteristic family must be 'plus' or 'minus'.")
    denominator = radius * M * math.cos(angle)
    if abs(denominator) < 1.0e-14:
        raise ArithmeticError("Axisymmetric compatibility source has a zero denominator.")
    return math.sin(theta) / denominator


def axis_compatibility_S(M: float, dtheta_dr: float) -> float:
    """Regular limit of S+/- at r = 0 for a smooth axisymmetric field."""
    if M <= 1.0:
        raise ValueError("The axis compatibility limit requires M > 1.")
    return dtheta_dr / math.sqrt(M * M - 1.0)
