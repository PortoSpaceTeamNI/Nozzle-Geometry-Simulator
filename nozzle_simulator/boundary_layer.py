"""BLIMP-inspired compressible axisymmetric boundary-layer marcher.

This is a reduced-order, non-reacting implementation of the BLIMP-J modelling
sequence, not a transcription of the historical JANNAF program.  It resolves a
wall-normal velocity profile, uses the two-layer Cebeci--Smith algebraic eddy
viscosity closure, and marches the thin-layer momentum equation downstream.
An adiabatic Walz/Crocco temperature relation closes the energy field, so the
gas-side wall temperature is predicted as the local recovery temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .flow import mach_from_area_ratio
from .models import (
    BoundaryLayerResult,
    CEAProperties,
    FlowResult,
    GeometryResult,
    NozzleInputs,
)

KAPPA = 0.40
VAN_DRIEST_A_PLUS = 26.0
CEBECI_SMITH_OUTER_COEFFICIENT = 0.0168


class BoundaryLayerSeparationError(ValueError):
    """Raised when the signed wall shear reaches zero in the divergent."""

    def __init__(self, axial_position_m: float, wall_shear_pa: float) -> None:
        self.axial_position_m = float(axial_position_m)
        self.wall_shear_pa = float(wall_shear_pa)
        super().__init__(
            "Boundary-layer separation detected at "
            f"x = {self.axial_position_m * 1e3:.3f} mm "
            f"(signed wall shear = {self.wall_shear_pa:.6g} Pa)."
        )


@dataclass(frozen=True)
class _EdgeState:
    wall_distance_m: np.ndarray
    radius_m: np.ndarray
    velocity_m_s: np.ndarray
    density_kg_m3: np.ndarray
    viscosity_pa_s: np.ndarray
    temperature_k: np.ndarray
    recovery_temperature_k: np.ndarray
    gas_constant_j_kg_k: np.ndarray


def _piecewise_station_property(
    x: np.ndarray,
    throat_x: float,
    chamber_value: float,
    throat_value: float,
    exit_value: float,
) -> np.ndarray:
    """Interpolate a CEA property on each side of the throat."""
    upstream = np.clip(x / max(throat_x, 1e-12), 0.0, 1.0)
    downstream = np.clip(
        (x - throat_x) / max(float(x[-1] - throat_x), 1e-12), 0.0, 1.0
    )
    return np.where(
        x <= throat_x,
        chamber_value + upstream * (throat_value - chamber_value),
        throat_value + downstream * (exit_value - throat_value),
    )


def _interpolate_viscosity(temperature: np.ndarray, cea: CEAProperties) -> np.ndarray:
    """Log-interpolate CEA viscosity and extrapolate mildly with a power law."""
    station_temperature = np.array(
        [cea.exit.temperature_k, cea.throat.temperature_k, cea.chamber.temperature_k]
    )
    station_viscosity = np.array(
        [
            cea.exit.viscosity_pa_s,
            cea.throat.viscosity_pa_s,
            cea.chamber.viscosity_pa_s,
        ]
    )
    if (
        not np.isfinite(station_temperature).all()
        or not np.isfinite(station_viscosity).all()
        or np.any(station_temperature <= 0.0)
        or np.any(station_viscosity <= 0.0)
    ):
        raise ArithmeticError(
            "CEA station temperatures and viscosities must be finite and positive."
        )
    order = np.argsort(station_temperature)
    temperature_safe = np.asarray(temperature, dtype=float)
    if not np.isfinite(temperature_safe).all() or np.any(temperature_safe <= 0.0):
        raise ArithmeticError("Gas temperature must be finite and positive.")
    log_mu = np.interp(
        np.log(temperature_safe),
        np.log(station_temperature[order]),
        np.log(station_viscosity[order]),
    )
    viscosity = np.exp(log_mu)
    below = temperature_safe < station_temperature[order][0]
    above = temperature_safe > station_temperature[order][-1]
    viscosity[below] = station_viscosity[order][0] * (
        temperature_safe[below] / station_temperature[order][0]
    ) ** 0.70
    viscosity[above] = station_viscosity[order][-1] * (
        temperature_safe[above] / station_temperature[order][-1]
    ) ** 0.70
    return viscosity


def adiabatic_wall_temperature(
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
) -> np.ndarray:
    """Return turbulent recovery temperature, with locally interpolated Prandtl."""
    prandtl = _piecewise_station_property(
        geometry.x_m,
        geometry.throat_x_m,
        cea.chamber.prandtl,
        cea.throat.prandtl,
        cea.exit.prandtl,
    )
    if not np.isfinite(prandtl).all() or np.any(prandtl <= 0.0):
        raise ArithmeticError("CEA Prandtl number must be finite and positive.")
    recovery_factor = prandtl ** (1.0 / 3.0)
    return flow.temperature_k * (
        1.0 + recovery_factor * 0.5 * (flow.gamma - 1.0) * flow.mach**2
    )


def _edge_state(
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
) -> _EdgeState:
    x = geometry.x_m
    wall_step = np.hypot(np.diff(x), np.diff(geometry.radius_m))
    wall_distance = np.concatenate(([0.0], np.cumsum(wall_step)))
    molecular_weight = _piecewise_station_property(
        x,
        geometry.throat_x_m,
        cea.chamber.molecular_weight_g_mol,
        cea.throat.molecular_weight_g_mol,
        cea.exit.molecular_weight_g_mol,
    ) / 1000.0
    gas_constant = 8.314462618 / molecular_weight
    viscosity = _interpolate_viscosity(flow.temperature_k, cea)
    velocity = flow.mach * np.sqrt(flow.gamma * gas_constant * flow.temperature_k)
    density = flow.pressure_bar * 1e5 / (gas_constant * flow.temperature_k)
    return _EdgeState(
        wall_distance_m=wall_distance,
        radius_m=geometry.radius_m,
        velocity_m_s=velocity,
        density_kg_m3=density,
        viscosity_pa_s=viscosity,
        temperature_k=flow.temperature_k,
        recovery_temperature_k=adiabatic_wall_temperature(geometry, flow, cea),
        gas_constant_j_kg_k=gas_constant,
    )


def _compressible_integral_thicknesses(
    y: np.ndarray,
    velocity: np.ndarray,
    density: np.ndarray,
    edge_velocity: float,
    edge_density: float,
) -> tuple[float, float]:
    if edge_velocity <= 0.0 or edge_density <= 0.0:
        raise ArithmeticError("Boundary-layer edge velocity and density must be positive.")
    velocity_ratio = velocity / edge_velocity
    mass_flux_ratio = density * velocity / (edge_density * edge_velocity)
    displacement = float(np.trapezoid(1.0 - mass_flux_ratio, y))
    momentum = float(np.trapezoid(mass_flux_ratio * (1.0 - velocity_ratio), y))
    return displacement, momentum


def _adiabatic_temperature_profile(
    velocity: np.ndarray,
    edge_velocity: float,
    edge_temperature: float,
    recovery_temperature: float,
) -> np.ndarray:
    """Adiabatic Walz relation; dT/dy=0 and T=Taw at the wall."""
    if edge_velocity <= 0.0:
        raise ArithmeticError("Boundary-layer edge velocity must be positive.")
    ratio = velocity / edge_velocity
    return recovery_temperature - (recovery_temperature - edge_temperature) * ratio**2


def _cebeci_smith_eddy_viscosity(
    y: np.ndarray,
    velocity: np.ndarray,
    density: np.ndarray,
    molecular_viscosity: np.ndarray,
    edge_velocity: float,
    edge_density: float,
    displacement_thickness: float,
    wall_shear: float,
) -> np.ndarray:
    """Two-layer Cebeci--Smith dynamic eddy viscosity for an attached wall layer."""
    gradient = np.gradient(velocity, y, edge_order=1)
    wall_density = float(density[0])
    wall_viscosity = float(molecular_viscosity[0])
    if wall_density <= 0.0 or wall_viscosity <= 0.0:
        raise ArithmeticError("Wall density and viscosity must be positive.")
    # u_tau is a turbulence velocity scale, so it uses the magnitude of tau_w.
    # The sign of tau_w itself is retained for separation detection.
    friction_velocity = math.sqrt(abs(wall_shear) / wall_density)
    y_plus = wall_density * friction_velocity * y / wall_viscosity
    mixing_length = KAPPA * y * (1.0 - np.exp(-y_plus / VAN_DRIEST_A_PLUS))
    inner = density * mixing_length**2 * np.abs(gradient)

    # Cebeci--Smith's outer wake scale with Klebanoff intermittency damping.
    if displacement_thickness <= 0.0:
        raise ArithmeticError(
            "The BLIMP-lite solution produced non-positive displacement thickness."
        )
    delta_scale = displacement_thickness
    klebanoff = 1.0 / (1.0 + 5.5 * (y / (6.0 * delta_scale)) ** 6)
    outer = (
        CEBECI_SMITH_OUTER_COEFFICIENT
        * edge_density
        * edge_velocity
        * delta_scale
        * klebanoff
    )
    eddy = np.minimum(inner, outer)
    eddy[0] = 0.0
    eddy[-1] = 0.0
    return eddy


def _initial_profile(
    y: np.ndarray,
    edge_velocity: float,
    initial_thickness: float,
) -> np.ndarray:
    eta = np.clip(y / max(initial_thickness, y[1]), 0.0, 1.0)
    # Smooth turbulent-like profile with exactly zero edge gradient.
    profile = eta ** (1.0 / 7.0)
    profile = profile + (1.0 - profile) * eta**8
    profile[0] = 0.0
    profile[eta >= 1.0] = 1.0
    return edge_velocity * profile


def _solve_station(
    y: np.ndarray,
    previous_velocity: np.ndarray,
    previous_density: np.ndarray,
    previous_edge_velocity: float,
    edge_velocity: float,
    edge_density: float,
    edge_temperature: float,
    recovery_temperature: float,
    edge_pressure_pa: float,
    gas_constant: float,
    cea: CEAProperties,
    streamwise_step: float,
    edge_acceleration: float,
    initial_wall_shear: float,
    previous_radius: float,
    radius: float,
    wall_axial_cosine: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Implicitly march one thin-layer momentum station with Picard iteration."""
    if previous_edge_velocity <= 0.0 or edge_velocity <= 0.0:
        raise ArithmeticError("Boundary-layer edge velocity must be positive.")
    if streamwise_step <= 0.0:
        raise ArithmeticError("Boundary-layer marching step must be positive.")
    velocity = previous_velocity * edge_velocity / previous_edge_velocity
    velocity[0], velocity[-1] = 0.0, edge_velocity
    wall_shear = float(initial_wall_shear)
    ds = streamwise_step

    for _ in range(12):
        temperature = _adiabatic_temperature_profile(
            velocity, edge_velocity, edge_temperature, recovery_temperature
        )
        if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
            raise ArithmeticError(
                "The BLIMP-lite velocity solution produced a non-physical temperature."
            )
        density = edge_pressure_pa / (gas_constant * temperature)
        viscosity = _interpolate_viscosity(temperature, cea)
        displacement, _ = _compressible_integral_thicknesses(
            y, velocity, density, edge_velocity, edge_density
        )
        eddy_viscosity = _cebeci_smith_eddy_viscosity(
            y,
            velocity,
            density,
            viscosity,
            edge_velocity,
            edge_density,
            displacement,
            wall_shear,
        )
        effective_viscosity = viscosity + eddy_viscosity

        radial_coordinate = radius - wall_axial_cosine * y
        previous_radial_coordinate = previous_radius - wall_axial_cosine * y
        if np.any(radial_coordinate <= 0.0) or np.any(previous_radial_coordinate <= 0.0):
            raise ArithmeticError("Boundary-layer grid crossed the nozzle axis.")
        streamwise_mass_gradient = (
            density * velocity * radial_coordinate
            - previous_density * previous_velocity * previous_radial_coordinate
        ) / ds
        cumulative_mass_gradient = np.zeros_like(y)
        cumulative_mass_gradient[1:] = np.cumsum(
            0.5
            * (streamwise_mass_gradient[:-1] + streamwise_mass_gradient[1:])
            * np.diff(y)
        )
        normal_velocity = -cumulative_mass_gradient / (density * radial_coordinate)

        count = y.size
        banded = np.zeros((3, count))
        right_hand_side = np.zeros(count)
        banded[1, 0] = 1.0
        banded[1, -1] = 1.0
        right_hand_side[-1] = edge_velocity
        for j in range(1, count - 1):
            dy_minus = y[j] - y[j - 1]
            dy_plus = y[j + 1] - y[j]
            control_width = 0.5 * (dy_minus + dy_plus)
            mu_minus = 0.5 * (effective_viscosity[j - 1] + effective_viscosity[j])
            mu_plus = 0.5 * (effective_viscosity[j] + effective_viscosity[j + 1])
            radius_minus = 0.5 * (
                radial_coordinate[j - 1] + radial_coordinate[j]
            )
            radius_plus = 0.5 * (
                radial_coordinate[j] + radial_coordinate[j + 1]
            )
            convection = density[j] * velocity[j] / ds
            lower = -radius_minus * mu_minus / (
                radial_coordinate[j] * dy_minus * control_width
            )
            upper = -radius_plus * mu_plus / (
                radial_coordinate[j] * dy_plus * control_width
            )
            diagonal = convection - lower - upper
            normal_convection = density[j] * normal_velocity[j]
            if normal_convection >= 0.0:
                diagonal += normal_convection / dy_minus
                lower -= normal_convection / dy_minus
            else:
                diagonal -= normal_convection / dy_plus
                upper += normal_convection / dy_plus
            banded[2, j - 1] = lower
            banded[1, j] = diagonal
            banded[0, j + 1] = upper
            right_hand_side[j] = (
                convection * previous_velocity[j]
                + density[j] * edge_velocity * edge_acceleration
            )
        solved = solve_banded((1, 1), banded, right_hand_side, check_finite=False)
        if not np.isfinite(solved).all():
            raise ArithmeticError("The BLIMP-lite velocity solve did not remain finite.")
        solved[0], solved[-1] = 0.0, edge_velocity
        updated_shear = float(
            viscosity[0] * (solved[1] - solved[0]) / (y[1] - y[0])
        )
        residual = float(np.max(np.abs(solved - velocity)) / max(edge_velocity, 1.0))
        velocity = 0.55 * solved + 0.45 * velocity
        wall_shear = 0.55 * updated_shear + 0.45 * wall_shear
        if residual < 2e-5:
            break

    temperature = _adiabatic_temperature_profile(
        velocity, edge_velocity, edge_temperature, recovery_temperature
    )
    if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
        raise ArithmeticError(
            "The BLIMP-lite velocity solution produced a non-physical temperature."
        )
    density = edge_pressure_pa / (gas_constant * temperature)
    viscosity = _interpolate_viscosity(temperature, cea)
    displacement, momentum = _compressible_integral_thicknesses(
        y, velocity, density, edge_velocity, edge_density
    )
    wall_shear = float(viscosity[0] * velocity[1] / (y[1] - y[0]))
    return velocity, temperature, density, wall_shear, displacement, momentum


def _march_profiles(
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
    edge: _EdgeState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """March a reduced set of stations, then interpolate to the geometry grid."""
    point_count = geometry.x_m.size
    march_count = min(point_count, 61)
    march_indices = np.unique(np.linspace(0, point_count - 1, march_count).astype(int))
    s = edge.wall_distance_m[march_indices]

    # The outer boundary is intentionally generous; the edge condition is imposed
    # there and all reported integrals are density weighted.
    max_length = max(float(s[-1]), 1e-4)
    y_extent = min(0.45 * float(np.min(geometry.radius_m)), max(0.018 * max_length, 0.0025))
    y = y_extent * np.linspace(0.0, 1.0, 41) ** 1.65

    displacement = np.zeros(march_indices.size)
    momentum = np.zeros(march_indices.size)
    wall_shear = np.zeros(march_indices.size)

    first = march_indices[0]
    initial_thickness = max(8.0 * y[1], 2.5e-4)
    velocity = _initial_profile(y, edge.velocity_m_s[first], initial_thickness)
    temperature = _adiabatic_temperature_profile(
        velocity,
        edge.velocity_m_s[first],
        edge.temperature_k[first],
        edge.recovery_temperature_k[first],
    )
    density = flow.pressure_bar[first] * 1e5 / (
        edge.gas_constant_j_kg_k[first] * temperature
    )
    viscosity = _interpolate_viscosity(temperature, cea)
    wall_shear[0] = float(viscosity[0] * velocity[1] / y[1])
    displacement[0], momentum[0] = _compressible_integral_thicknesses(
        y,
        velocity,
        density,
        edge.velocity_m_s[first],
        edge.density_kg_m3[first],
    )

    for local_index in range(1, march_indices.size):
        index = march_indices[local_index]
        previous_index = march_indices[local_index - 1]
        ds = s[local_index] - s[local_index - 1]
        if ds <= 0.0:
            raise ArithmeticError("Boundary-layer marching step must be positive.")
        acceleration = (
            edge.velocity_m_s[index] - edge.velocity_m_s[previous_index]
        ) / ds
        wall_axial_cosine = abs(
            (geometry.x_m[index] - geometry.x_m[previous_index])
            / ds
        )
        velocity, _, density, wall_shear[local_index], displacement[local_index], momentum[
            local_index
        ] = _solve_station(
            y=y,
            previous_velocity=velocity,
            previous_density=density,
            previous_edge_velocity=edge.velocity_m_s[previous_index],
            edge_velocity=edge.velocity_m_s[index],
            edge_density=edge.density_kg_m3[index],
            edge_temperature=edge.temperature_k[index],
            recovery_temperature=edge.recovery_temperature_k[index],
            edge_pressure_pa=flow.pressure_bar[index] * 1e5,
            gas_constant=edge.gas_constant_j_kg_k[index],
            cea=cea,
            streamwise_step=ds,
            edge_acceleration=acceleration,
            initial_wall_shear=wall_shear[local_index - 1],
            previous_radius=geometry.radius_m[previous_index],
            radius=geometry.radius_m[index],
            wall_axial_cosine=wall_axial_cosine,
        )
        if (
            geometry.x_m[index] >= geometry.throat_x_m
            and wall_shear[local_index] <= 0.0
        ):
            raise BoundaryLayerSeparationError(
                geometry.x_m[index], wall_shear[local_index]
            )

    full_s = edge.wall_distance_m
    return tuple(
        np.interp(full_s, s, values)
        for values in (displacement, momentum, wall_shear)
    )


def _quick_shape_factor(
    flow: FlowResult,
    recovery_temperature: np.ndarray,
) -> np.ndarray:
    """Density-weighted H from an imposed 1/7-power/Walz profile.

    This intentionally weak closure exists only for fast GA screening.  The full
    simulation and the high-fidelity optimization path use the profile marcher.
    """
    eta = np.linspace(0.0, 1.0, 129)
    velocity_ratio = eta ** (1.0 / 7.0)
    edge_temperature = flow.temperature_k[:, None]
    wall_temperature = recovery_temperature[:, None]
    temperature = wall_temperature - (
        wall_temperature - edge_temperature
    ) * velocity_ratio[None, :] ** 2
    density_ratio = edge_temperature / np.maximum(temperature, 50.0)
    displacement_ratio = np.trapezoid(
        1.0 - density_ratio * velocity_ratio[None, :], eta, axis=1
    )
    momentum_ratio = np.trapezoid(
        density_ratio
        * velocity_ratio[None, :]
        * (1.0 - velocity_ratio[None, :]),
        eta,
        axis=1,
    )
    return displacement_ratio / np.maximum(momentum_ratio, 1e-12)


def _integrate_quick_momentum_thickness(
    wall_distance: np.ndarray,
    radius: np.ndarray,
    velocity: np.ndarray,
    mach: np.ndarray,
    shape_factor: np.ndarray,
    skin_friction: np.ndarray,
) -> np.ndarray:
    """March the compressible axisymmetric von Karman integral equation."""
    velocity_gradient = np.gradient(velocity, wall_distance, edge_order=1)
    radius_gradient = np.gradient(radius, wall_distance, edge_order=1)
    coefficient = (
        radius_gradient / np.maximum(radius, 1e-12)
        + (shape_factor + 2.0 - mach**2)
        * velocity_gradient
        / np.maximum(velocity, 1e-9)
    )
    momentum = np.zeros_like(wall_distance)
    for index in range(1, wall_distance.size):
        step = wall_distance[index] - wall_distance[index - 1]
        local_coefficient = 0.5 * (coefficient[index - 1] + coefficient[index])
        source = 0.25 * (skin_friction[index - 1] + skin_friction[index])
        exponent = float(np.clip(-local_coefficient * step, -50.0, 50.0))
        if abs(local_coefficient * step) < 1e-8:
            next_value = momentum[index - 1] + source * step
        else:
            decay = math.exp(exponent)
            next_value = (
                momentum[index - 1] * decay
                + source * (1.0 - decay) / local_coefficient
            )
        momentum[index] = max(float(next_value), 0.0)
    return momentum


def _effective_mach_and_velocity_efficiency(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
    effective_radius: np.ndarray,
) -> tuple[np.ndarray, float]:
    throat_index = int(np.argmin(np.abs(geometry.x_m - geometry.throat_x_m)))
    if not np.isfinite(effective_radius).all() or np.any(effective_radius <= 0.0):
        raise ArithmeticError(
            "Boundary-layer displacement closed or inverted the effective flow area."
        )
    effective_throat_radius = effective_radius[throat_index]
    effective_area_ratio = (effective_radius / effective_throat_radius) ** 2
    if np.any(effective_area_ratio < 1.0 - 1e-10):
        raise ArithmeticError(
            "The effective minimum area moved away from the geometric throat."
        )
    # Only remove round-off below the exact A/A*=1 throat condition.
    effective_area_ratio[effective_area_ratio < 1.0] = 1.0
    mach_effective = np.array(
        [
            mach_from_area_ratio(area_ratio, gamma, bool(x >= geometry.throat_x_m))
            for area_ratio, gamma, x in zip(
                effective_area_ratio, flow.gamma, geometry.x_m
            )
        ]
    )
    gamma_exit = cea.exit.gamma
    gas_constant_exit = 8.314462618 / (cea.exit.molecular_weight_g_mol / 1000.0)
    ideal_temperature = cea.chamber.temperature_k / (
        1.0 + 0.5 * (gamma_exit - 1.0) * flow.mach[-1] ** 2
    )
    effective_temperature = cea.chamber.temperature_k / (
        1.0 + 0.5 * (gamma_exit - 1.0) * mach_effective[-1] ** 2
    )
    ideal_velocity = flow.mach[-1] * math.sqrt(
        gamma_exit * gas_constant_exit * ideal_temperature
    )
    effective_velocity = mach_effective[-1] * math.sqrt(
        gamma_exit * gas_constant_exit * effective_temperature
    )
    if ideal_velocity <= 0.0:
        raise ArithmeticError("Ideal exit velocity must be positive.")
    efficiency = float(effective_velocity / ideal_velocity)
    if not np.isfinite(efficiency):
        raise ArithmeticError("Boundary-layer velocity efficiency is not finite.")
    return mach_effective, efficiency


def compute_quick_boundary_layer(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
) -> BoundaryLayerResult:
    """Fast weak-closure boundary layer for GA screening only.

    It uses an Eckert reference-temperature flat-plate skin-friction correlation
    and a one-equation momentum-integral march.  It is deliberately exposed as a
    lower-fidelity alternative rather than presented as BLIMP-equivalent physics.
    """
    edge = _edge_state(geometry, flow, cea)
    reference_length = np.maximum(edge.wall_distance_m, 1e-6)
    recovery_temperature = edge.recovery_temperature_k
    reference_temperature = edge.temperature_k + 0.72 * (
        recovery_temperature - edge.temperature_k
    )
    reference_viscosity = _interpolate_viscosity(reference_temperature, cea)
    reference_density = flow.pressure_bar * 1e5 / (
        edge.gas_constant_j_kg_k * np.maximum(reference_temperature, 50.0)
    )
    reynolds_reference = np.maximum(
        reference_density
        * edge.velocity_m_s
        * reference_length
        / reference_viscosity,
        1e3,
    )
    skin_friction = 0.0592 / reynolds_reference**0.2
    shape_factor = _quick_shape_factor(flow, recovery_temperature)
    momentum = _integrate_quick_momentum_thickness(
        edge.wall_distance_m,
        geometry.radius_m,
        edge.velocity_m_s,
        flow.mach,
        shape_factor,
        skin_friction,
    )
    displacement = shape_factor * momentum
    wall_shear = 0.5 * skin_friction * edge.density_kg_m3 * edge.velocity_m_s**2
    effective_radius = np.maximum(geometry.radius_m - displacement, 1e-6)
    mach_effective, velocity_efficiency = _effective_mach_and_velocity_efficiency(
        inputs, geometry, flow, cea, effective_radius
    )
    return BoundaryLayerResult(
        displacement_thickness_m=displacement,
        momentum_thickness_m=momentum,
        shape_factor=shape_factor,
        skin_friction_coefficient=skin_friction,
        wall_shear_stress_pa=wall_shear,
        wall_temperature_k=recovery_temperature,
        effective_radius_m=effective_radius,
        reynolds=reynolds_reference,
        viscosity_pa_s=edge.viscosity_pa_s,
        mach_effective=mach_effective,
        velocity_efficiency=velocity_efficiency,
    )


def compute_boundary_layer(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    flow: FlowResult,
    cea: CEAProperties,
) -> BoundaryLayerResult:
    """Solve the attached, turbulent, adiabatic BLIMP-lite boundary layer."""
    edge = _edge_state(geometry, flow, cea)
    displacement, momentum, wall_shear = _march_profiles(
        geometry, flow, cea, edge
    )
    separated = np.flatnonzero(
        (geometry.x_m >= geometry.throat_x_m) & (wall_shear <= 0.0)
    )
    if separated.size:
        index = int(separated[0])
        raise BoundaryLayerSeparationError(
            geometry.x_m[index], wall_shear[index]
        )
    # The adiabatic boundary condition is local and known exactly on the full grid;
    # do not retain interpolation error from the reduced marching grid.
    wall_temperature = edge.recovery_temperature_k.copy()
    if np.any(momentum <= 0.0):
        raise ArithmeticError(
            "The BLIMP-lite solution produced non-positive momentum thickness."
        )
    shape_factor = displacement / momentum
    dynamic_pressure_twice = edge.density_kg_m3 * edge.velocity_m_s**2
    if np.any(dynamic_pressure_twice <= 0.0):
        raise ArithmeticError("Boundary-layer edge dynamic pressure must be positive.")
    skin_friction = 2.0 * wall_shear / dynamic_pressure_twice
    effective_radius = geometry.radius_m - displacement
    if np.any(effective_radius <= 0.0):
        raise ArithmeticError(
            "Boundary-layer displacement closed or inverted the effective flow area."
        )
    mach_effective, velocity_efficiency = _effective_mach_and_velocity_efficiency(
        inputs, geometry, flow, cea, effective_radius
    )
    reynolds = (
        edge.density_kg_m3
        * edge.velocity_m_s
        * edge.wall_distance_m
        / edge.viscosity_pa_s
    )
    return BoundaryLayerResult(
        displacement_thickness_m=displacement,
        momentum_thickness_m=momentum,
        shape_factor=shape_factor,
        skin_friction_coefficient=skin_friction,
        wall_shear_stress_pa=wall_shear,
        wall_temperature_k=wall_temperature,
        effective_radius_m=effective_radius,
        reynolds=reynolds,
        viscosity_pa_s=edge.viscosity_pa_s,
        mach_effective=mach_effective,
        velocity_efficiency=velocity_efficiency,
    )
