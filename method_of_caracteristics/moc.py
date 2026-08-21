"""Pressure-based axisymmetric MOC analysis of a prescribed bell contour."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from nozzle_simulator.models import CEAProperties, GeometryResult, MOCResult, NozzleInputs

from .initial_transient_line import (
    KliegelLevineInitialLine,
    SauerInitialLine,
    build_kliegel_levine_initial_line,
    build_sauer_initial_line,
)
from .numerical_methods import characteristic_slope, odd_axis_gradient
from .physical_properties import (
    area_ratio_from_mach,
    axis_compatibility_S,
    compatibility_Q,
    compatibility_S,
    pressure_from_mach,
    state_from_pressure,
)

SAUER_PROJECTED_AXIS_MACH = 1.05
MAX_KL_TRANSITION_RADIAL_STATIONS = 161


@dataclass(frozen=True)
class MOCSettings:
    axial_stations: int = 360
    radial_stations: int = 41
    initialization: str = "kliegel_levine"
    start_mach: float = 1.12
    corrector_iterations: int = 40
    tolerance: float = 2.0e-8

    def validate(self) -> None:
        if self.axial_stations < 20:
            raise ValueError("MOC axial_stations must be at least 20.")
        if self.radial_stations < 7:
            raise ValueError("MOC radial_stations must be at least 7.")
        if self.initialization not in {"kliegel_levine", "sauer", "quasi_1d"}:
            raise ValueError(
                "MOC initialization must be 'kliegel_levine', 'sauer' or "
                "'quasi_1d'."
            )
        if self.start_mach <= 1.0:
            raise ValueError("MOC start_mach must be greater than one.")
        if self.corrector_iterations < 2:
            raise ValueError("MOC corrector_iterations must be at least two.")
        if self.tolerance <= 0.0:
            raise ValueError("MOC tolerance must be positive.")


@dataclass(frozen=True)
class _GasModel:
    stagnation_pressure_pa: float
    stagnation_temperature_k: float
    gamma: float
    gas_constant_j_kg_k: float

    def state(self, pressure_pa: float) -> tuple[float, float, float, float]:
        state = state_from_pressure(
            pressure_pa,
            self.stagnation_pressure_pa,
            self.stagnation_temperature_k,
            self.gamma,
            self.gas_constant_j_kg_k,
        )
        if state[3] <= 1.0:
            raise ValueError(
                "The characteristic update produced a non-supersonic state. "
                "Refine the axial mesh or move the initial line farther downstream."
            )
        return state


@dataclass(frozen=True)
class _FootState:
    radius: float
    pressure: float
    theta: float
    M: float
    density: float
    velocity: float
    Q: float
    S: float
    delta_x: float


def _wall_interpolator(geometry: GeometryResult) -> PchipInterpolator:
    mask = geometry.x_m >= geometry.throat_x_m - 1.0e-12
    x = np.asarray(geometry.x_m[mask], dtype=float)
    radius = np.asarray(geometry.radius_m[mask], dtype=float)
    order = np.argsort(x)
    x, radius = x[order], radius[order]
    _rounded_x, unique = np.unique(np.round(x, 13), return_index=True)
    x, radius = x[unique], radius[unique]
    if x.size < 4 or np.any(np.diff(x) <= 0.0):
        raise ValueError("The divergent contour is not suitable for MOC interpolation.")
    return PchipInterpolator(x, radius, extrapolate=False)


def _interpolate(radius: np.ndarray, values: np.ndarray, location: float) -> float:
    return float(np.interp(location, radius, values))


def _trace_foot(
    *,
    target_x: float,
    target_radius: float,
    old_x: np.ndarray,
    old_radius: np.ndarray,
    old_pressure: np.ndarray,
    old_theta: np.ndarray,
    old_M: np.ndarray,
    old_density: np.ndarray,
    old_velocity: np.ndarray,
    target_M: float,
    target_theta: float,
    family: str,
    tolerance: float,
) -> _FootState:
    old_x = np.asarray(old_x, dtype=float)
    if old_x.shape != old_radius.shape:
        raise ValueError("The upstream-boundary x and radius arrays must have equal shape.")
    target_slope = characteristic_slope(target_M, target_theta, family)
    lower, upper = float(old_radius[0]), float(old_radius[-1])
    radial_scale = max(upper - lower, 1.0e-12)

    def residual(foot_radius: float) -> float:
        foot_M = _interpolate(old_radius, old_M, foot_radius)
        foot_theta = _interpolate(old_radius, old_theta, foot_radius)
        foot_slope = characteristic_slope(foot_M, foot_theta, family)
        foot_x = _interpolate(old_radius, old_x, foot_radius)
        delta_x = target_x - foot_x
        return (
            foot_radius
            + 0.5 * delta_x * (foot_slope + target_slope)
            - target_radius
        )

    same_radius = min(max(target_radius, lower), upper)
    local_dx = target_x - _interpolate(old_radius, old_x, same_radius)
    foot_radius = target_radius - local_dx * target_slope
    fixed_point_converged = False
    for _ in range(30):
        if not math.isfinite(foot_radius) or foot_radius < lower or foot_radius > upper:
            break
        foot_M = _interpolate(old_radius, old_M, foot_radius)
        foot_theta = _interpolate(old_radius, old_theta, foot_radius)
        foot_slope = characteristic_slope(foot_M, foot_theta, family)
        foot_x = _interpolate(old_radius, old_x, foot_radius)
        updated = target_radius - 0.5 * (target_x - foot_x) * (
            foot_slope + target_slope
        )
        if abs(updated - foot_radius) <= tolerance * radial_scale:
            foot_radius = updated
            fixed_point_converged = lower <= foot_radius <= upper
            break
        foot_radius = 0.45 * foot_radius + 0.55 * updated

    if fixed_point_converged:
        physical_roots = [float(foot_radius)]
    else:
        samples = np.linspace(lower, upper, max(2 * old_radius.size, 48))
        residuals = np.array([residual(float(value)) for value in samples])
        roots: list[float] = []
        root_tolerance = tolerance * radial_scale
        for index in range(samples.size - 1):
            left, right = float(samples[index]), float(samples[index + 1])
            f_left, f_right = float(residuals[index]), float(residuals[index + 1])
            if abs(f_left) <= root_tolerance:
                roots.append(left)
            if f_left * f_right < 0.0:
                roots.append(float(brentq(residual, left, right)))
        if abs(float(residuals[-1])) <= root_tolerance:
            roots.append(float(samples[-1]))
        if not roots:
            raise ValueError(
                f"The backward {family} characteristic did not intersect the upstream "
                "boundary. Increase axial/radial resolution."
            )
        physical_roots = [
            root
            for root in roots
            if target_x - _interpolate(old_radius, old_x, root) >= -1.0e-12
        ]
    if not physical_roots:
        raise ValueError(f"The {family} characteristic intersection lies downstream.")
    foot_radius = min(
        physical_roots,
        key=lambda root: target_x - _interpolate(old_radius, old_x, root),
    )
    foot_x = _interpolate(old_radius, old_x, foot_radius)
    delta_x = target_x - foot_x
    pressure = _interpolate(old_radius, old_pressure, foot_radius)
    theta = _interpolate(old_radius, old_theta, foot_radius)
    M = _interpolate(old_radius, old_M, foot_radius)
    density = _interpolate(old_radius, old_density, foot_radius)
    velocity = _interpolate(old_radius, old_velocity, foot_radius)
    Q = compatibility_Q(M, density, velocity)
    if foot_radius <= 1.0e-12:
        dtheta_dr = odd_axis_gradient(old_radius, old_theta)
        S = axis_compatibility_S(M, dtheta_dr)
    else:
        S = compatibility_S(M, theta, foot_radius, family)
    return _FootState(foot_radius, pressure, theta, M, density, velocity, Q, S, delta_x)


def _interior_point(
    *,
    target_x: float,
    target_radius: float,
    old_x: np.ndarray,
    old_radius: np.ndarray,
    old_pressure: np.ndarray,
    old_theta: np.ndarray,
    old_M: np.ndarray,
    old_density: np.ndarray,
    old_velocity: np.ndarray,
    pressure_guess: float,
    theta_guess: float,
    gas: _GasModel,
    settings: MOCSettings,
) -> tuple[float, float]:
    pressure, theta = float(pressure_guess), float(theta_guess)
    for iteration in range(settings.corrector_iterations):
        if not math.isfinite(pressure) or not math.isfinite(theta):
            raise ArithmeticError(
                f"Non-finite interior predictor at iteration {iteration}: "
                f"p={pressure!r}, theta={theta!r}, r={target_radius!r}."
            )
        _temperature, density, velocity, M = gas.state(pressure)
        foot_minus = _trace_foot(
            target_x=target_x,
            target_radius=target_radius,
            old_x=old_x,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            target_M=M,
            target_theta=theta,
            family="minus",
            tolerance=settings.tolerance,
        )
        foot_plus = _trace_foot(
            target_x=target_x,
            target_radius=target_radius,
            old_x=old_x,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            target_M=M,
            target_theta=theta,
            family="plus",
            tolerance=settings.tolerance,
        )
        Q_P = compatibility_Q(M, density, velocity)
        S_P_minus = compatibility_S(M, theta, target_radius, "minus")
        S_P_plus = compatibility_S(M, theta, target_radius, "plus")
        Q_minus = 0.5 * (foot_minus.Q + Q_P)
        Q_plus = 0.5 * (foot_plus.Q + Q_P)
        S_minus = 0.5 * (foot_minus.S + S_P_minus)
        S_plus = 0.5 * (foot_plus.S + S_P_plus)

        rhs_minus = (
            Q_minus * foot_minus.pressure
            - foot_minus.theta
            - S_minus * foot_minus.delta_x
        )
        rhs_plus = (
            Q_plus * foot_plus.pressure
            + foot_plus.theta
            - S_plus * foot_plus.delta_x
        )
        pressure_new = (rhs_minus + rhs_plus) / (Q_minus + Q_plus)
        theta_new = rhs_plus - Q_plus * pressure_new
        if not math.isfinite(pressure_new) or not math.isfinite(theta_new):
            raise ArithmeticError(
                f"Non-finite interior correction at iteration {iteration}: "
                f"p={pressure_new!r}, theta={theta_new!r}, "
                f"Q-= {Q_minus!r}, Q+= {Q_plus!r}."
            )
        gas.state(pressure_new)
        error = max(
            abs(pressure_new - pressure) / max(abs(pressure_new), 1.0),
            abs(theta_new - theta),
        )
        pressure = 0.35 * pressure + 0.65 * pressure_new
        theta = 0.35 * theta + 0.65 * theta_new
        if error < settings.tolerance:
            return float(pressure_new), float(theta_new)
    raise RuntimeError("The interior MOC predictor-corrector did not converge.")


def _wall_point(
    *,
    target_x: float,
    target_radius: float,
    wall_theta: float,
    old_x: np.ndarray,
    old_radius: np.ndarray,
    old_pressure: np.ndarray,
    old_theta: np.ndarray,
    old_M: np.ndarray,
    old_density: np.ndarray,
    old_velocity: np.ndarray,
    pressure_guess: float,
    gas: _GasModel,
    settings: MOCSettings,
) -> float:
    pressure = float(pressure_guess)
    for _ in range(settings.corrector_iterations):
        _temperature, density, velocity, M = gas.state(pressure)
        foot_plus = _trace_foot(
            target_x=target_x,
            target_radius=target_radius,
            old_x=old_x,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            target_M=M,
            target_theta=wall_theta,
            family="plus",
            tolerance=settings.tolerance,
        )
        Q_P = compatibility_Q(M, density, velocity)
        S_P_plus = compatibility_S(M, wall_theta, target_radius, "plus")
        Q_plus = 0.5 * (foot_plus.Q + Q_P)
        S_plus = 0.5 * (foot_plus.S + S_P_plus)
        pressure_new = foot_plus.pressure - (
            wall_theta - foot_plus.theta + S_plus * foot_plus.delta_x
        ) / Q_plus
        gas.state(pressure_new)
        error = abs(pressure_new - pressure) / max(abs(pressure_new), 1.0)
        pressure = 0.35 * pressure + 0.65 * pressure_new
        if error < settings.tolerance:
            return float(pressure_new)
    raise RuntimeError("The wall MOC predictor-corrector did not converge.")


def _axis_point(
    *,
    target_x: float,
    old_x: np.ndarray,
    new_radius: np.ndarray,
    new_theta: np.ndarray,
    old_radius: np.ndarray,
    old_pressure: np.ndarray,
    old_theta: np.ndarray,
    old_M: np.ndarray,
    old_density: np.ndarray,
    old_velocity: np.ndarray,
    pressure_guess: float,
    gas: _GasModel,
    settings: MOCSettings,
) -> float:
    pressure = float(pressure_guess)
    dtheta_dr = odd_axis_gradient(new_radius, new_theta)
    for _ in range(settings.corrector_iterations):
        _temperature, density, velocity, M = gas.state(pressure)
        foot_minus = _trace_foot(
            target_x=target_x,
            target_radius=0.0,
            old_x=old_x,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            target_M=M,
            target_theta=0.0,
            family="minus",
            tolerance=settings.tolerance,
        )
        Q_P = compatibility_Q(M, density, velocity)
        S_axis_minus = axis_compatibility_S(M, dtheta_dr)
        Q_minus = 0.5 * (foot_minus.Q + Q_P)
        S_minus = 0.5 * (foot_minus.S + S_axis_minus)
        pressure_new = foot_minus.pressure - (
            foot_minus.theta + S_minus * foot_minus.delta_x
        ) / Q_minus
        gas.state(pressure_new)
        error = abs(pressure_new - pressure) / max(abs(pressure_new), 1.0)
        pressure = 0.35 * pressure + 0.65 * pressure_new
        if error < settings.tolerance:
            return float(pressure_new)
    raise RuntimeError("The axis MOC predictor-corrector did not converge.")


@dataclass(frozen=True)
class _PlaneState:
    x: float
    radius: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    M: np.ndarray
    theta: np.ndarray


def _resample_plane_radially(
    plane: _PlaneState,
    *,
    radial_stations: int,
    gas: _GasModel,
) -> _PlaneState:
    """Interpolate a solved plane onto a different body-fitted radial grid."""
    if radial_stations == plane.radius.size:
        return plane
    target_radius = np.linspace(0.0, float(plane.radius[-1]), radial_stations)
    pressure = np.interp(target_radius, plane.radius, plane.pressure)
    theta = np.interp(target_radius, plane.radius, plane.theta)
    theta[0] = 0.0
    theta[-1] = float(plane.theta[-1])
    temperature, density, velocity, mach = _states_from_pressure(pressure, gas)
    return _PlaneState(
        x=plane.x,
        radius=target_radius,
        pressure=pressure,
        temperature=temperature,
        density=density,
        velocity=velocity,
        M=mach,
        theta=theta,
    )


@dataclass(frozen=True)
class _CharacteristicNode:
    """One state in the curved characteristic transition net."""

    x: float
    radius: float
    pressure: float
    theta: float
    temperature: float
    density: float
    velocity: float
    M: float


@dataclass(frozen=True)
class _CharacteristicLine:
    """One C- line ordered from the nozzle wall to the symmetry axis."""

    nodes: tuple[_CharacteristicNode, ...]

    @property
    def wall_x(self) -> float:
        return self.nodes[0].x

    @property
    def axis_x(self) -> float:
        return self.nodes[-1].x


class _UpstreamTransitionIntersection(ValueError):
    """A C+ ray was already consumed before the next transition C- line."""


def _make_node(
    *, x: float, radius: float, pressure: float, theta: float, gas: _GasModel
) -> _CharacteristicNode:
    temperature, density, velocity, M = gas.state(float(pressure))
    return _CharacteristicNode(
        x=float(x),
        radius=float(radius),
        pressure=float(pressure),
        theta=float(theta),
        temperature=float(temperature),
        density=float(density),
        velocity=float(velocity),
        M=float(M),
    )


def _node_Q(node: _CharacteristicNode) -> float:
    return compatibility_Q(node.M, node.density, node.velocity)


def _intersect_two_characteristics(
    *,
    minus: _CharacteristicNode,
    plus: _CharacteristicNode,
    plus_axis_gradient: float | None = None,
    gas: _GasModel,
    settings: MOCSettings,
) -> _CharacteristicNode:
    """Intersect C- from ``minus`` with C+ from ``plus``."""
    pressure = 0.5 * (minus.pressure + plus.pressure)
    theta = 0.5 * (minus.theta + plus.theta)
    x = max(minus.x, plus.x) + 1.0e-7
    radius = 0.5 * (minus.radius + plus.radius)
    for _ in range(settings.corrector_iterations):
        point = _make_node(
            x=x, radius=max(radius, 1.0e-10), pressure=pressure, theta=theta, gas=gas
        )
        slope_minus = 0.5 * (
            characteristic_slope(minus.M, minus.theta, "minus")
            + characteristic_slope(point.M, point.theta, "minus")
        )
        slope_plus = 0.5 * (
            characteristic_slope(plus.M, plus.theta, "plus")
            + characteristic_slope(point.M, point.theta, "plus")
        )
        denominator = slope_minus - slope_plus
        if abs(denominator) < 1.0e-12:
            raise ArithmeticError("Transition characteristics became parallel.")
        x_new = (
            plus.radius
            - minus.radius
            + slope_minus * minus.x
            - slope_plus * plus.x
        ) / denominator
        radius_new = minus.radius + slope_minus * (x_new - minus.x)
        if x_new <= max(minus.x, plus.x) or radius_new <= 0.0:
            raise _UpstreamTransitionIntersection(
                "Transition characteristic intersection is not downstream."
            )

        point = _make_node(
            x=x_new,
            radius=radius_new,
            pressure=pressure,
            theta=theta,
            gas=gas,
        )
        Q_minus = 0.5 * (_node_Q(minus) + _node_Q(point))
        Q_plus = 0.5 * (_node_Q(plus) + _node_Q(point))
        S_minus = 0.5 * (
            compatibility_S(minus.M, minus.theta, minus.radius, "minus")
            + compatibility_S(point.M, point.theta, point.radius, "minus")
        )
        plus_source = (
            axis_compatibility_S(plus.M, plus_axis_gradient)
            if plus.radius <= 1.0e-12 and plus_axis_gradient is not None
            else compatibility_S(plus.M, plus.theta, plus.radius, "plus")
        )
        S_plus = 0.5 * (
            plus_source
            + compatibility_S(point.M, point.theta, point.radius, "plus")
        )
        rhs_minus = (
            Q_minus * minus.pressure
            - minus.theta
            - S_minus * (x_new - minus.x)
        )
        rhs_plus = (
            Q_plus * plus.pressure
            + plus.theta
            - S_plus * (x_new - plus.x)
        )
        pressure_new = (rhs_minus + rhs_plus) / (Q_minus + Q_plus)
        theta_new = rhs_plus - Q_plus * pressure_new
        gas.state(pressure_new)
        error = max(
            abs(pressure_new - pressure) / max(abs(pressure_new), 1.0),
            abs(theta_new - theta),
            abs(x_new - x) / max(abs(x_new), 1.0e-6),
            abs(radius_new - radius) / max(abs(radius_new), 1.0e-6),
        )
        pressure = 0.35 * pressure + 0.65 * pressure_new
        theta = 0.35 * theta + 0.65 * theta_new
        x = 0.35 * x + 0.65 * x_new
        radius = 0.35 * radius + 0.65 * radius_new
        if error < settings.tolerance:
            return _make_node(
                x=x_new,
                radius=radius_new,
                pressure=pressure_new,
                theta=theta_new,
                gas=gas,
            )
    raise RuntimeError("Transition interior-point corrector did not converge.")


def _find_wall_intersection_x(
    *,
    plus: _CharacteristicNode,
    point_M: float,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
) -> float:
    """Locate the wall intersection of a C+ characteristic."""
    lower = plus.x + 1.0e-10
    upper = float(wall.x[-1])
    slope_plus = characteristic_slope(plus.M, plus.theta, "plus")

    def residual(x_value: float) -> float:
        theta_wall = math.atan(float(wall_derivative(x_value)))
        point_slope = characteristic_slope(point_M, theta_wall, "plus")
        characteristic_radius = plus.radius + 0.5 * (
            slope_plus + point_slope
        ) * (x_value - plus.x)
        return float(wall(x_value)) - characteristic_radius

    samples = np.linspace(lower, upper, 300)
    values = np.array([residual(float(value)) for value in samples])
    for index in range(samples.size - 1):
        if values[index] == 0.0:
            return float(samples[index])
        if values[index] * values[index + 1] < 0.0:
            return float(
                brentq(residual, float(samples[index]), float(samples[index + 1]))
            )
    raise ValueError("The transition C+ characteristic did not reach the nozzle wall.")


def _transition_wall_point(
    *,
    plus: _CharacteristicNode,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gas: _GasModel,
    settings: MOCSettings,
) -> _CharacteristicNode:
    pressure = plus.pressure
    x = plus.x
    for _ in range(settings.corrector_iterations):
        _temperature, density, velocity, M = gas.state(pressure)
        x_new = _find_wall_intersection_x(
            plus=plus,
            point_M=M,
            wall=wall,
            wall_derivative=wall_derivative,
        )
        radius_new = float(wall(x_new))
        theta_new = math.atan(float(wall_derivative(x_new)))
        point = _CharacteristicNode(
            x=x_new,
            radius=radius_new,
            pressure=pressure,
            theta=theta_new,
            temperature=_temperature,
            density=density,
            velocity=velocity,
            M=M,
        )
        Q_plus = 0.5 * (_node_Q(plus) + _node_Q(point))
        S_plus = 0.5 * (
            compatibility_S(plus.M, plus.theta, plus.radius, "plus")
            + compatibility_S(point.M, point.theta, point.radius, "plus")
        )
        pressure_new = plus.pressure - (
            theta_new - plus.theta + S_plus * (x_new - plus.x)
        ) / Q_plus
        gas.state(pressure_new)
        error = max(
            abs(pressure_new - pressure) / max(abs(pressure_new), 1.0),
            abs(x_new - x) / max(abs(x_new), 1.0e-6),
        )
        pressure = 0.35 * pressure + 0.65 * pressure_new
        x = 0.35 * x + 0.65 * x_new
        if error < settings.tolerance:
            return _make_node(
                x=x_new,
                radius=radius_new,
                pressure=pressure_new,
                theta=theta_new,
                gas=gas,
            )
    raise RuntimeError("Transition wall-point corrector did not converge.")


def _transition_axis_point(
    *,
    minus: _CharacteristicNode,
    near_axis_nodes: list[_CharacteristicNode],
    gas: _GasModel,
    settings: MOCSettings,
) -> _CharacteristicNode:
    pressure = minus.pressure
    x = minus.x
    near_axis = list(reversed(near_axis_nodes[-2:]))
    radii = np.array([0.0] + [node.radius for node in near_axis])
    angles = np.array([0.0] + [node.theta for node in near_axis])
    order = np.argsort(radii)
    dtheta_dr = odd_axis_gradient(radii[order], angles[order])
    for _ in range(settings.corrector_iterations):
        temperature, density, velocity, M = gas.state(pressure)
        slope_minus = 0.5 * (
            characteristic_slope(minus.M, minus.theta, "minus")
            + characteristic_slope(M, 0.0, "minus")
        )
        x_new = minus.x - minus.radius / slope_minus
        if x_new <= minus.x:
            raise ValueError("Transition axis intersection is not downstream.")
        point = _CharacteristicNode(
            x=x_new,
            radius=0.0,
            pressure=pressure,
            theta=0.0,
            temperature=temperature,
            density=density,
            velocity=velocity,
            M=M,
        )
        Q_minus = 0.5 * (_node_Q(minus) + _node_Q(point))
        S_minus = 0.5 * (
            compatibility_S(minus.M, minus.theta, minus.radius, "minus")
            + axis_compatibility_S(M, dtheta_dr)
        )
        pressure_new = minus.pressure - (
            minus.theta + S_minus * (x_new - minus.x)
        ) / Q_minus
        try:
            gas.state(pressure_new)
        except ValueError as exc:
            raise ValueError(
                "Transition axis state is unphysical: "
                f"pA={minus.pressure:.6g}, pP={pressure_new:.6g}, "
                f"rA={minus.radius:.6g}, dx={x_new-minus.x:.6g}, "
                f"thetaA={minus.theta:.6g}, S={S_minus:.6g}, Q={Q_minus:.6g}."
            ) from exc
        error = max(
            abs(pressure_new - pressure) / max(abs(pressure_new), 1.0),
            abs(x_new - x) / max(abs(x_new), 1.0e-6),
        )
        pressure = 0.35 * pressure + 0.65 * pressure_new
        x = 0.35 * x + 0.65 * x_new
        if error < settings.tolerance:
            return _make_node(
                x=x_new,
                radius=0.0,
                pressure=pressure_new,
                theta=0.0,
                gas=gas,
            )
    raise RuntimeError("Transition axis-point corrector did not converge.")


def _initial_characteristic_line(
    initial_line: KliegelLevineInitialLine, gas: _GasModel
) -> _CharacteristicLine:
    nodes = tuple(
        _make_node(
            x=float(initial_line.x_m[index]),
            radius=float(initial_line.radius_m[index]),
            pressure=float(initial_line.pressure_pa[index]),
            theta=float(initial_line.theta_rad[index]),
            gas=gas,
        )
        for index in range(initial_line.radius_m.size - 1, -1, -1)
    )
    return _CharacteristicLine(nodes=nodes)


def _next_transition_line(
    *,
    previous: _CharacteristicLine,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gas: _GasModel,
    settings: MOCSettings,
    wall_node: _CharacteristicNode | None = None,
    stop_x: float | None = None,
) -> _CharacteristicLine:
    if len(previous.nodes) < 3:
        raise ValueError("At least three points are required on a transition line.")
    nodes: list[_CharacteristicNode] = [
        wall_node
        if wall_node is not None
        else _transition_wall_point(
            plus=previous.nodes[1],
            wall=wall,
            wall_derivative=wall_derivative,
            gas=gas,
            settings=settings,
        )
    ]
    previous_near_axis = list(reversed(previous.nodes[-3:]))
    previous_radii = np.array([node.radius for node in previous_near_axis])
    previous_angles = np.array([node.theta for node in previous_near_axis])
    previous_order = np.argsort(previous_radii)
    previous_axis_gradient = odd_axis_gradient(
        previous_radii[previous_order], previous_angles[previous_order]
    )
    skipped_upstream = 0
    for index in range(2, len(previous.nodes)):
        try:
            node = _intersect_two_characteristics(
                minus=nodes[-1],
                plus=previous.nodes[index],
                plus_axis_gradient=(
                    previous_axis_gradient
                    if previous.nodes[index].radius <= 1.0e-12
                    else None
                ),
                gas=gas,
                settings=settings,
            )
        except _UpstreamTransitionIntersection:
            # At high radial resolution the wall intersection may lie beyond
            # one or more closely spaced C+ rays. Those rays have left the
            # physical domain through the wall and must not seed an upstream
            # intersection on the new C- line.
            skipped_upstream += 1
            continue
        nodes.append(node)
        if stop_x is not None and nodes[-1].x >= stop_x:
            return _CharacteristicLine(nodes=tuple(nodes))
    if len(nodes) < 2:
        raise RuntimeError(
            "Transition line contains no downstream interior point after "
            f"skipping {skipped_upstream} consumed C+ rays."
        )
    nodes.append(
        _transition_axis_point(
            minus=nodes[-1],
            near_axis_nodes=nodes,
            gas=gas,
            settings=settings,
        )
    )
    return _CharacteristicLine(nodes=tuple(nodes))


def _sample_transition_plane(
    *,
    lines: list[_CharacteristicLine],
    target_x: float,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gas: _GasModel,
    radial_stations: int,
) -> _PlaneState:
    samples: list[_CharacteristicNode] = []
    for line in lines:
        x_values = np.array([node.x for node in line.nodes])
        if target_x < np.min(x_values) - 1.0e-12 or target_x > np.max(x_values) + 1.0e-12:
            continue
        order = np.argsort(x_values)
        x_ordered = x_values[order]
        unique_x, unique = np.unique(np.round(x_ordered, 13), return_index=True)
        if unique_x.size < 2:
            continue
        selected = np.asarray(order)[unique]

        def field(
            name: str,
            current_line: _CharacteristicLine = line,
            current_selected: np.ndarray = selected,
            current_x: np.ndarray = unique_x,
        ) -> float:
            values = np.array(
                [
                    getattr(current_line.nodes[index], name)
                    for index in current_selected
                ]
            )
            return float(np.interp(target_x, current_x, values))

        samples.append(
            _make_node(
                x=target_x,
                radius=field("radius"),
                pressure=field("pressure"),
                theta=field("theta"),
                gas=gas,
            )
        )

    wall_nodes = [line.nodes[0] for line in lines]
    wall_x = np.array([node.x for node in wall_nodes])
    wall_pressure = np.array([node.pressure for node in wall_nodes])
    wall_order = np.argsort(wall_x)
    wall_pressure_at_target = float(
        np.interp(target_x, wall_x[wall_order], wall_pressure[wall_order])
    )
    samples.append(
        _make_node(
            x=target_x,
            radius=float(wall(target_x)),
            pressure=wall_pressure_at_target,
            theta=math.atan(float(wall_derivative(target_x))),
            gas=gas,
        )
    )
    samples.sort(key=lambda node: node.radius)
    sample_radius = np.array([node.radius for node in samples])
    rounded_radius, unique = np.unique(np.round(sample_radius, 12), return_index=True)
    samples = [samples[index] for index in unique]
    sample_radius = rounded_radius
    if sample_radius.size < 5 or sample_radius[0] > 1.0e-9:
        raise RuntimeError("Transition net did not cover the complete vertical section.")
    target_radius = np.linspace(0.0, float(wall(target_x)), radial_stations)
    pressure = np.interp(
        target_radius, sample_radius, [node.pressure for node in samples]
    )
    theta = np.interp(target_radius, sample_radius, [node.theta for node in samples])
    theta[0] = 0.0
    theta[-1] = math.atan(float(wall_derivative(target_x)))
    temperature, density, velocity, mach = _states_from_pressure(pressure, gas)
    return _PlaneState(
        x=float(target_x),
        radius=target_radius,
        pressure=pressure,
        temperature=temperature,
        density=density,
        velocity=velocity,
        M=mach,
        theta=theta,
    )


def _bootstrap_kliegel_levine_line(
    *,
    initial_line: KliegelLevineInitialLine,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gas: _GasModel,
    settings: MOCSettings,
) -> tuple[_PlaneState, tuple[_CharacteristicLine, ...]]:
    """Fill the characteristic wedge before the first complete x-plane."""
    lines = [_initial_characteristic_line(initial_line, gas)]
    target_x = lines[0].axis_x
    max_lines = max(4 * settings.radial_stations, 80)
    while lines[-1].wall_x < target_x and len(lines) < max_lines:
        try:
            next_wall = _transition_wall_point(
                plus=lines[-1].nodes[1],
                wall=wall,
                wall_derivative=wall_derivative,
                gas=gas,
                settings=settings,
            )
            if next_wall.x >= target_x:
                lines.append(_CharacteristicLine(nodes=(next_wall,)))
                break
            lines.append(
                _next_transition_line(
                    previous=lines[-1],
                    wall=wall,
                    wall_derivative=wall_derivative,
                    gas=gas,
                    settings=settings,
                    wall_node=next_wall,
                    stop_x=target_x,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Kliegel-Levine transition failed while constructing C- line "
                f"{len(lines) + 1}; previous wall/axis x = "
                f"{lines[-1].wall_x:.6g}/{lines[-1].axis_x:.6g} m."
            ) from exc
    if lines[-1].wall_x < target_x:
        raise RuntimeError("Transition net did not reach the first complete axial plane.")
    return (
        _sample_transition_plane(
            lines=lines,
            target_x=target_x,
            wall=wall,
            wall_derivative=wall_derivative,
            gas=gas,
            radial_stations=settings.radial_stations,
        ),
        tuple(lines),
    )


def _states_from_pressure(
    pressure: np.ndarray, gas: _GasModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temperature = np.empty_like(pressure)
    density = np.empty_like(pressure)
    velocity = np.empty_like(pressure)
    mach = np.empty_like(pressure)
    for column, pressure_value in enumerate(pressure):
        state = gas.state(float(pressure_value))
        temperature[column], density[column], velocity[column], mach[column] = state
    return temperature, density, velocity, mach


def _project_sauer_to_full_axial_plane(
    *,
    initial_line: SauerInitialLine,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gamma: float,
    gas: _GasModel,
    radial_stations: int,
) -> _PlaneState:
    """Sample Sauer's analytic field on the first wholly supersonic x-plane.

    The pressure-based marcher is body-fitted to constant-x planes.  The
    original curved Sauer line is retained in the result for diagnostics, while
    this projection supplies a topology-compatible first plane.
    """
    target_critical_velocity_ratio = SAUER_PROJECTED_AXIS_MACH * math.sqrt(
        (gamma + 1.0)
        / (2.0 + (gamma - 1.0) * SAUER_PROJECTED_AXIS_MACH**2)
    )
    local_x = (target_critical_velocity_ratio - 1.0) / initial_line.alpha_1_m
    axis_sonic_x = float(np.max(initial_line.x_m) - initial_line.eta_m)
    target_x = axis_sonic_x + local_x
    wall_radius = float(wall(target_x))
    radius = np.linspace(0.0, wall_radius, radial_stations)
    alpha = initial_line.alpha_1_m

    u_perturbation = (
        alpha * local_x + 0.25 * (gamma + 1.0) * alpha**2 * radius**2
    )
    v_perturbation = (
        0.5 * (gamma + 1.0) * alpha**2 * local_x * radius
        + ((gamma + 1.0) ** 2 / 16.0) * alpha**3 * radius**3
    )
    axial_ratio = 1.0 + u_perturbation
    critical_velocity_ratio = np.hypot(axial_ratio, v_perturbation)
    sound_speed_ratio_squared = (
        0.5 * (gamma + 1.0)
        - 0.5 * (gamma - 1.0) * critical_velocity_ratio**2
    )
    if np.any(sound_speed_ratio_squared <= 0.0):
        raise ValueError("The projected Sauer plane has a non-positive sound speed.")
    mach = critical_velocity_ratio / np.sqrt(sound_speed_ratio_squared)
    if np.any(mach <= 1.0):
        raise ValueError("The projected Sauer axial plane is not wholly supersonic.")
    pressure = gas.stagnation_pressure_pa * (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    ) ** (-gamma / (gamma - 1.0))
    temperature, density, velocity, recovered_mach = _states_from_pressure(pressure, gas)
    theta = np.arctan2(v_perturbation, axial_ratio)
    theta[0] = 0.0
    theta[-1] = math.atan(float(wall_derivative(target_x)))
    return _PlaneState(
        x=target_x,
        radius=radius,
        pressure=pressure,
        temperature=temperature,
        density=density,
        velocity=velocity,
        M=recovered_mach,
        theta=theta,
    )


def _march_to_plane(
    *,
    target_x: float,
    target_radius: np.ndarray,
    old_x: np.ndarray,
    old_radius: np.ndarray,
    old_pressure: np.ndarray,
    old_theta: np.ndarray,
    old_M: np.ndarray,
    old_density: np.ndarray,
    old_velocity: np.ndarray,
    wall_theta: float,
    gas: _GasModel,
    settings: MOCSettings,
    inner_pressure: float | None = None,
    inner_theta: float = 0.0,
) -> _PlaneState:
    """March from a general upstream curve to one constant-x plane."""
    columns = target_radius.size
    pressure = np.full(columns, np.nan)
    theta = np.full(columns, np.nan)

    for column in range(1, columns - 1):
        same_fraction_radius = min(
            max(float(target_radius[column]), float(old_radius[0])),
            float(old_radius[-1]),
        )
        pressure_guess = _interpolate(old_radius, old_pressure, same_fraction_radius)
        theta_guess = _interpolate(old_radius, old_theta, same_fraction_radius)
        pressure[column], theta[column] = _interior_point(
            target_x=target_x,
            target_radius=float(target_radius[column]),
            old_x=old_x,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            pressure_guess=pressure_guess,
            theta_guess=theta_guess,
            gas=gas,
            settings=settings,
        )

    theta[-1] = wall_theta
    pressure[-1] = _wall_point(
        target_x=target_x,
        target_radius=float(target_radius[-1]),
        wall_theta=wall_theta,
        old_x=old_x,
        old_radius=old_radius,
        old_pressure=old_pressure,
        old_theta=old_theta,
        old_M=old_M,
        old_density=old_density,
        old_velocity=old_velocity,
        pressure_guess=float(old_pressure[-1]),
        gas=gas,
        settings=settings,
    )

    if inner_pressure is not None:
        pressure[0] = inner_pressure
        theta[0] = inner_theta
    else:
        theta[0] = 0.0
        pressure[0] = _axis_point(
            target_x=target_x,
            old_x=old_x,
            new_radius=target_radius,
            new_theta=theta,
            old_radius=old_radius,
            old_pressure=old_pressure,
            old_theta=old_theta,
            old_M=old_M,
            old_density=old_density,
            old_velocity=old_velocity,
            pressure_guess=float(old_pressure[0]),
            gas=gas,
            settings=settings,
        )

    temperature, density, velocity, mach = _states_from_pressure(pressure, gas)
    return _PlaneState(
        x=float(target_x),
        radius=np.asarray(target_radius, dtype=float),
        pressure=pressure,
        temperature=temperature,
        density=density,
        velocity=velocity,
        M=mach,
        theta=theta,
    )


def _sauer_upstream_boundary(
    initial_line: SauerInitialLine,
    new_inner_radius: float,
    previous: _PlaneState | None,
    radial_stations: int,
) -> tuple[np.ndarray, ...]:
    """Join the newly exposed Sauer segment to the preceding axial plane."""
    if previous is None:
        upper_radius = float(initial_line.radius_m[-1])
        segment_count = radial_stations
    else:
        upper_radius = float(previous.radius[0])
        fraction = (upper_radius - new_inner_radius) / initial_line.radius_m[-1]
        segment_count = max(4, math.ceil(fraction * (radial_stations - 1)) + 1)

    segment_radius = np.linspace(new_inner_radius, upper_radius, segment_count)
    (
        segment_x,
        segment_pressure,
        _segment_temperature,
        segment_density,
        segment_velocity,
        _segment_axial_velocity,
        _segment_radial_velocity,
        segment_M,
        segment_theta,
    ) = initial_line.state_at_radius(segment_radius)

    if previous is None:
        return (
            segment_x,
            segment_radius,
            segment_pressure,
            segment_theta,
            segment_M,
            segment_density,
            segment_velocity,
        )

    return (
        np.concatenate((segment_x[:-1], np.full(previous.radius.size, previous.x))),
        np.concatenate((segment_radius[:-1], previous.radius)),
        np.concatenate((segment_pressure[:-1], previous.pressure)),
        np.concatenate((segment_theta[:-1], previous.theta)),
        np.concatenate((segment_M[:-1], previous.M)),
        np.concatenate((segment_density[:-1], previous.density)),
        np.concatenate((segment_velocity[:-1], previous.velocity)),
    )


def _bootstrap_sauer_line(
    *,
    initial_line: SauerInitialLine,
    wall: PchipInterpolator,
    wall_derivative: PchipInterpolator,
    gas: _GasModel,
    settings: MOCSettings,
    progress: Callable[[int, int], None] | None,
    progress_total: int,
) -> tuple[_PlaneState, int]:
    """Fill the non-simple region between the curved line and the axis plane."""
    rt = float(initial_line.radius_m[-1])
    divergent_span = float(wall.x[-1] - wall.x[0])
    curved_span = float(np.max(initial_line.x_m) - np.min(initial_line.x_m))
    bootstrap_planes = max(
        32,
        min(
            max(settings.radial_stations * 2, 20),
            math.ceil(settings.axial_stations * curved_span / divergent_span),
        ),
    )
    previous: _PlaneState | None = None
    for step in range(1, bootstrap_planes + 1):
        inner_radius = rt * (1.0 - step / bootstrap_planes)
        (
            target_x_array,
            inner_pressure_array,
            _inner_temperature,
            _inner_density,
            _inner_velocity,
            _inner_axial_velocity,
            _inner_radial_velocity,
            _inner_M,
            inner_theta_array,
        ) = initial_line.state_at_radius(inner_radius)
        target_x = float(target_x_array)
        wall_radius = float(wall(target_x))
        if not math.isfinite(wall_radius) or wall_radius <= inner_radius:
            raise ValueError("The Sauer bootstrap produced an invalid annular plane.")
        target_radius = np.linspace(inner_radius, wall_radius, settings.radial_stations)
        upstream = _sauer_upstream_boundary(
            initial_line, inner_radius, previous, settings.radial_stations
        )
        try:
            previous = _march_to_plane(
                target_x=target_x,
                target_radius=target_radius,
                old_x=upstream[0],
                old_radius=upstream[1],
                old_pressure=upstream[2],
                old_theta=upstream[3],
                old_M=upstream[4],
                old_density=upstream[5],
                old_velocity=upstream[6],
                wall_theta=math.atan(float(wall_derivative(target_x))),
                gas=gas,
                settings=settings,
                inner_pressure=float(inner_pressure_array),
                inner_theta=float(inner_theta_array),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Sauer bootstrap failed at plane {step}/{bootstrap_planes}, "
                f"inner radius {inner_radius:.6g} m."
            ) from exc
        if progress is not None and (step == bootstrap_planes or step % 2 == 0):
            progress(step, progress_total)
    if previous is None or abs(float(previous.radius[0])) > 1.0e-12:
        raise RuntimeError("The Sauer bootstrap did not reach the symmetry axis.")
    return previous, bootstrap_planes


def analyze_prescribed_nozzle(
    inputs: NozzleInputs,
    geometry: GeometryResult,
    cea: CEAProperties,
    *,
    friction_thrust_coefficient: float = 0.0,
    settings: MOCSettings | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MOCResult:
    """March a smooth, shock-free axisymmetric field through the prescribed wall."""
    total_started = time.perf_counter()
    settings = settings or MOCSettings()
    settings.validate()
    wall = _wall_interpolator(geometry)
    wall_derivative = wall.derivative()

    gamma = float(cea.throat.gamma)
    molecular_weight_kg_mol = cea.throat.molecular_weight_g_mol / 1000.0
    gas_constant = 8.314462618 / molecular_weight_kg_mol
    gas = _GasModel(
        stagnation_pressure_pa=inputs.chamber_pressure_bar * 1.0e5,
        stagnation_temperature_k=cea.chamber.temperature_k,
        gamma=gamma,
        gas_constant_j_kg_k=gas_constant,
    )
    throat_x = max(float(geometry.throat_x_m), float(wall.x[0]))
    exit_x = min(float(geometry.exit_x_m), float(wall.x[-1]))

    initialization_started = time.perf_counter()
    sauer_line: SauerInitialLine | None = None
    kliegel_levine_line: KliegelLevineInitialLine | None = None
    transition_line_count = 0
    bootstrap_count = 0
    if settings.initialization == "kliegel_levine":
        transition_radial_stations = min(
            settings.radial_stations,
            MAX_KL_TRANSITION_RADIAL_STATIONS,
        )
        transition_settings = replace(
            settings,
            radial_stations=transition_radial_stations,
        )
        kliegel_levine_line = build_kliegel_levine_initial_line(
            stagnation_pressure_pa=gas.stagnation_pressure_pa,
            stagnation_temperature_k=gas.stagnation_temperature_k,
            gamma=gamma,
            gas_constant_j_kg_k=gas_constant,
            throat_radius_m=inputs.throat_radius_m,
            throat_x_m=throat_x,
            radial_stations=transition_radial_stations,
        )
        first_plane, transition_lines = _bootstrap_kliegel_levine_line(
            initial_line=kliegel_levine_line,
            wall=wall,
            wall_derivative=wall_derivative,
            gas=gas,
            settings=transition_settings,
        )
        first_plane = _resample_plane_radially(
            first_plane,
            radial_stations=settings.radial_stations,
            gas=gas,
        )
        transition_line_count = len(transition_lines)
        transition_line_x = tuple(
            np.array([node.x for node in line.nodes]) for line in transition_lines
        )
        transition_line_radius = tuple(
            np.array([node.radius for node in line.nodes]) for line in transition_lines
        )
        progress_total = settings.axial_stations
        start_x = first_plane.x
        initialization_label = (
            "Kliegel-Levine third-order curved line with characteristic transition "
            f"({transition_line_count} C- lines)"
        )
        reported_start_mach = float(np.min(first_plane.M))
        initial_line_x = kliegel_levine_line.x_m
        initial_line_radius = kliegel_levine_line.radius_m
        initial_line_mach = kliegel_levine_line.mach
        initial_line_theta = kliegel_levine_line.theta_rad
        initial_line_pressure = kliegel_levine_line.pressure_pa
        sauer_alpha = float("nan")
        sauer_eta = float("nan")
    elif settings.initialization == "sauer":
        transition_line_x = ()
        transition_line_radius = ()
        sauer_line = build_sauer_initial_line(
            stagnation_pressure_pa=gas.stagnation_pressure_pa,
            stagnation_temperature_k=gas.stagnation_temperature_k,
            gamma=gamma,
            gas_constant_j_kg_k=gas_constant,
            throat_radius_m=inputs.throat_radius_m,
            throat_x_m=throat_x,
            radial_stations=settings.radial_stations,
        )
        first_plane = _project_sauer_to_full_axial_plane(
            initial_line=sauer_line,
            wall=wall,
            wall_derivative=wall_derivative,
            gamma=gamma,
            gas=gas,
            radial_stations=settings.radial_stations,
        )
        progress_total = settings.axial_stations
        start_x = first_plane.x
        initialization_label = (
            "Sauer transonic field projected at axis M = 1.05 "
            "(Rc,sub = 1.5 Rt)"
        )
        reported_start_mach = float(np.min(first_plane.M))
        initial_line_x = sauer_line.x_m
        initial_line_radius = sauer_line.radius_m
        initial_line_mach = sauer_line.mach
        initial_line_theta = sauer_line.theta_rad
        initial_line_pressure = sauer_line.pressure_pa
        sauer_alpha = sauer_line.alpha_1_m
        sauer_eta = sauer_line.eta_m
    else:
        transition_line_x = ()
        transition_line_radius = ()
        target_area_ratio = area_ratio_from_mach(settings.start_mach, gamma)
        target_radius = inputs.throat_radius_m * math.sqrt(target_area_ratio)
        if target_radius >= geometry.exit_radius_m:
            raise ValueError("The selected MOC start Mach lies beyond the nozzle exit area.")
        start_x = float(
            brentq(
                lambda axial: float(wall(axial)) - target_radius,
                throat_x + 1.0e-12,
                exit_x,
            )
        )
        first_radius = float(wall(start_x)) * np.linspace(
            0.0, 1.0, settings.radial_stations
        )
        first_pressure = np.full(
            settings.radial_stations,
            pressure_from_mach(settings.start_mach, gas.stagnation_pressure_pa, gamma),
        )
        first_theta = math.atan(float(wall_derivative(start_x))) * np.linspace(
            0.0, 1.0, settings.radial_stations
        )
        first_temperature, first_density, first_velocity, first_M = _states_from_pressure(
            first_pressure, gas
        )
        first_plane = _PlaneState(
            x=start_x,
            radius=first_radius,
            pressure=first_pressure,
            temperature=first_temperature,
            density=first_density,
            velocity=first_velocity,
            M=first_M,
            theta=first_theta,
        )
        initialization_label = "Quasi-1D isentropic reference line"
        reported_start_mach = settings.start_mach
        initial_line_x = np.full(settings.radial_stations, start_x)
        initial_line_radius = first_radius.copy()
        initial_line_mach = first_M.copy()
        initial_line_theta = first_theta.copy()
        initial_line_pressure = first_pressure.copy()
        sauer_alpha = float("nan")
        sauer_eta = float("nan")
        progress_total = settings.axial_stations

    if start_x >= exit_x:
        raise ValueError("The MOC initial-data construction lies beyond the nozzle exit.")
    initialization_time = time.perf_counter() - initialization_started

    axial_progress = np.linspace(0.0, 1.0, settings.axial_stations)
    x = start_x + (exit_x - start_x) * axial_progress**1.18
    x[0], x[-1] = start_x, exit_x
    radial_fraction = np.linspace(0.0, 1.0, settings.radial_stations)
    wall_radius = np.asarray(wall(x), dtype=float)
    wall_radius[0] = float(wall(start_x))
    wall_radius[-1] = float(wall(exit_x))
    if not np.isfinite(wall_radius).all():
        raise ArithmeticError("The prescribed wall interpolation produced non-finite radii.")
    radius = wall_radius[:, None] * radial_fraction[None, :]

    shape = radius.shape
    pressure = np.full(shape, np.nan)
    temperature = np.full(shape, np.nan)
    density = np.full(shape, np.nan)
    velocity = np.full(shape, np.nan)
    M = np.full(shape, np.nan)
    theta = np.full(shape, np.nan)

    radius[0] = first_plane.radius
    pressure[0] = first_plane.pressure
    temperature[0] = first_plane.temperature
    density[0] = first_plane.density
    velocity[0] = first_plane.velocity
    M[0] = first_plane.M
    theta[0] = first_plane.theta

    marching_started = time.perf_counter()
    for row in range(1, settings.axial_stations):
        old_radius = radius[row - 1]
        new_radius = radius[row]
        previous_fields = {
            "radius": old_radius,
            "pressure": pressure[row - 1],
            "theta": theta[row - 1],
            "Mach": M[row - 1],
            "density": density[row - 1],
            "velocity": velocity[row - 1],
        }
        invalid_fields = [
            name for name, values in previous_fields.items() if not np.isfinite(values).all()
        ]
        if invalid_fields:
            raise ArithmeticError(
                f"Non-finite MOC data entered axial row {row}: {', '.join(invalid_fields)}."
            )

        try:
            plane = _march_to_plane(
                target_x=float(x[row]),
                target_radius=new_radius,
                old_x=np.full(old_radius.size, float(x[row - 1])),
                old_radius=old_radius,
                old_pressure=pressure[row - 1],
                old_theta=theta[row - 1],
                old_M=M[row - 1],
                old_density=density[row - 1],
                old_velocity=velocity[row - 1],
                wall_theta=math.atan(float(wall_derivative(float(x[row])))),
                gas=gas,
                settings=settings,
            )
        except Exception as exc:
            raise RuntimeError(f"MOC axial plane failed at row {row}.") from exc
        pressure[row] = plane.pressure
        theta[row] = plane.theta
        temperature[row] = plane.temperature
        density[row] = plane.density
        velocity[row] = plane.velocity
        M[row] = plane.M
        if progress is not None and (row == settings.axial_stations - 1 or row % 4 == 0):
            progress(bootstrap_count + row + 1, progress_total)

    marching_time = time.perf_counter() - marching_started

    axial_velocity = velocity * np.cos(theta)
    mass_flow = np.array(
        [
            2.0
            * math.pi
            * np.trapezoid(density[row] * axial_velocity[row] * radius[row], radius[row])
            for row in range(settings.axial_stations)
        ]
    )
    mass_flow_residual = float(abs(mass_flow[-1] - mass_flow[0]) / mass_flow[0])
    throat_area = math.pi * inputs.throat_radius_m**2
    cea_choked_mass_flow = (
        gas.stagnation_pressure_pa * throat_area / float(cea.cstar_m_s)
    )
    initial_mass_flow_error = float(
        abs(mass_flow[0] - cea_choked_mass_flow) / cea_choked_mass_flow
    )
    ambient_pressure_pa = inputs.ambient_pressure_bar * 1.0e5
    exit_integrand = (
        density[-1] * axial_velocity[-1] ** 2 + pressure[-1] - ambient_pressure_pa
    ) * radius[-1]
    inviscid_thrust = float(2.0 * math.pi * np.trapezoid(exit_integrand, radius[-1]))
    inviscid_cf = inviscid_thrust / (gas.stagnation_pressure_pa * throat_area)
    # Preserve the signed BLIMP wall-friction result.  Clamping it here would
    # conceal an upstream sign error or an invalid boundary-layer solution.
    friction_corrected_cf = inviscid_cf - float(friction_thrust_coefficient)

    warnings: list[str] = []
    if settings.initialization == "kliegel_levine":
        warnings.append(
            "Kliegel-Levine uses a constant-gamma perfect-gas throat expansion. The "
            "curved line is connected to the fixed-x grid by a solved characteristic "
            "transition net; verify the reported mass-flow residual before using Cf."
        )
        if settings.radial_stations > MAX_KL_TRANSITION_RADIAL_STATIONS:
            warnings.append(
                "The Kliegel-Levine transition net was solved with "
                f"Nr={MAX_KL_TRANSITION_RADIAL_STATIONS} and interpolated onto the "
                f"Nr={settings.radial_stations} downstream grid. This preserves the "
                "validated transonic topology while refining the prescribed-wall march."
            )
    elif settings.initialization == "sauer":
        warnings.append(
            "Sauer is projected from its curved characteristic line onto the fixed-x "
            "marching grid. That projection changes the integrated mass flow, so the "
            "resulting thrust coefficient is diagnostic only and must not be used for "
            "geometry ranking."
        )
    else:
        warnings.append(
            "The reference initial line is quasi-one-dimensional and not a transonic throat solution."
        )
    if mass_flow_residual > 5.0e-3:
        warnings.append(
            f"Mass-flow residual {mass_flow_residual:.3%} exceeds the 0.5% verification target."
        )
    if initial_mass_flow_error > 5.0e-3:
        warnings.append(
            "Initial MOC mass flow differs from the CEA choked value by "
            f"{initial_mass_flow_error:.3%}; the solution must not be used for "
            "geometry ranking."
        )
    return MOCResult(
        x_m=x,
        radial_fraction=radial_fraction,
        radius_m=radius,
        pressure_pa=pressure,
        temperature_k=temperature,
        density_kg_m3=density,
        velocity_m_s=velocity,
        mach=M,
        theta_rad=theta,
        axial_velocity_m_s=axial_velocity,
        mass_flow_kg_s=mass_flow,
        mass_flow_residual=mass_flow_residual,
        inviscid_thrust_n=inviscid_thrust,
        inviscid_thrust_coefficient=float(inviscid_cf),
        friction_corrected_thrust_coefficient=float(friction_corrected_cf),
        start_mach=reported_start_mach,
        gamma=gamma,
        gas_constant_j_kg_k=gas_constant,
        converged=True,
        initialization=initialization_label,
        initial_line_x_m=np.asarray(initial_line_x, dtype=float),
        initial_line_radius_m=np.asarray(initial_line_radius, dtype=float),
        initial_line_mach=np.asarray(initial_line_mach, dtype=float),
        initial_line_theta_rad=np.asarray(initial_line_theta, dtype=float),
        initial_line_pressure_pa=np.asarray(initial_line_pressure, dtype=float),
        transition_line_x_m=transition_line_x,
        transition_line_radius_m=transition_line_radius,
        sauer_alpha_1_m=float(sauer_alpha),
        sauer_eta_m=float(sauer_eta),
        initialization_time_s=float(initialization_time),
        marching_time_s=float(marching_time),
        total_time_s=float(time.perf_counter() - total_started),
        cea_choked_mass_flow_kg_s=float(cea_choked_mass_flow),
        initial_mass_flow_error=float(initial_mass_flow_error),
        warnings=tuple(warnings),
    )
