"""Ambient nozzle-performance closure used by simulation and optimization."""

import math

import numpy as np

from .models import (
    BoundaryLayerResult,
    CEAProperties,
    GeometryResult,
    NozzleInputs,
    PerformanceResult,
)


def compute_performance(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    cea: CEAProperties,
    boundary_layer: BoundaryLayerResult,
) -> PerformanceResult:
    """Combine CEA momentum thrust with divergence, BL and ambient pressure thrust."""
    pressure_ratio = cea.exit_pressure_bar / inputs.chamber_pressure_bar
    ambient_ratio = inputs.ambient_pressure_bar / inputs.chamber_pressure_bar
    momentum_cf = cea.ideal_momentum_thrust_coefficient
    divergence_efficiency = 0.5 * (
        1.0 + math.cos(math.radians(geometry.theta_out_deg))
    )
    momentum_efficiency = divergence_efficiency * boundary_layer.velocity_efficiency
    pressure_cf = inputs.expansion_ratio * (pressure_ratio - ambient_ratio)
    throat_area = math.pi * inputs.throat_radius_m**2
    # Axial projection of wall shear: dFx = tau_w (dx/ds) 2 pi r ds.
    friction_force = float(
        np.trapezoid(
            2.0 * math.pi * geometry.radius_m * boundary_layer.wall_shear_stress_pa,
            geometry.x_m,
        )
    )
    friction_cf = friction_force / (
        inputs.chamber_pressure_bar * 1e5 * throat_area
    )
    effective_cf = momentum_efficiency * momentum_cf + pressure_cf - friction_cf
    effective_thrust = effective_cf * inputs.chamber_pressure_bar * 1e5 * throat_area
    return PerformanceResult(
        divergence_efficiency=float(divergence_efficiency),
        momentum_efficiency=float(momentum_efficiency),
        momentum_thrust_coefficient=float(momentum_cf),
        pressure_thrust_coefficient=float(pressure_cf),
        friction_thrust_coefficient=float(friction_cf),
        effective_thrust_coefficient=float(effective_cf),
        effective_thrust_n=float(effective_thrust),
    )


def performance_fitness(performance: PerformanceResult, ambient_mode: str) -> float:
    """Reject separated operation; otherwise maximize effective ambient thrust coefficient."""
    if "Separated" in ambient_mode:
        return 0.0
    value = performance.effective_thrust_coefficient
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def loss_breakdown(performance: PerformanceResult) -> dict[str, float]:
    """Return non-negative modeled loss contributions in thrust-coefficient units.

    Divergence and blockage are attributed sequentially so that their sum equals
    the complete modeled reduction of the ideal CEA momentum contribution.  A
    positive pressure term is a gain and is therefore excluded from loss shares.
    """
    momentum_cf = performance.momentum_thrust_coefficient
    divergence = max(
        (1.0 - performance.divergence_efficiency) * momentum_cf, 0.0
    )
    blockage = max(
        (
            performance.divergence_efficiency
            - performance.momentum_efficiency
        )
        * momentum_cf,
        0.0,
    )
    return {
        "Exit divergence": divergence,
        "BL displacement": blockage,
        "Wall friction": max(performance.friction_thrust_coefficient, 0.0),
        "Ambient mismatch": max(-performance.pressure_thrust_coefficient, 0.0),
    }
