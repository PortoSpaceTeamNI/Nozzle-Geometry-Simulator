"""Sauer low-order axisymmetric transonic initial-data line."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SAUER_SUBSONIC_CURVATURE_RATIO = 1.5


@dataclass(frozen=True)
class SauerInitialLine:
    """Complete flow state on the curved Sauer initial-data line."""

    x_m: np.ndarray
    radius_m: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    axial_velocity_m_s: np.ndarray
    radial_velocity_m_s: np.ndarray
    mach: np.ndarray
    theta_rad: np.ndarray
    alpha_1_m: float
    eta_m: float
    critical_speed_m_s: float
    subsonic_curvature_radius_m: float

    def state_at_radius(
        self, radius_m: np.ndarray | float
    ) -> tuple[np.ndarray, ...]:
        """Interpolate every initial-line field at one or more radii."""
        radius = np.asarray(radius_m, dtype=float)
        fields = (
            self.x_m,
            self.pressure_pa,
            self.temperature_k,
            self.density_kg_m3,
            self.velocity_m_s,
            self.axial_velocity_m_s,
            self.radial_velocity_m_s,
            self.mach,
            self.theta_rad,
        )
        return tuple(np.interp(radius, self.radius_m, field) for field in fields)


@dataclass(frozen=True)
class KliegelLevineInitialLine:
    """Third-order Kliegel--Levine state on a curved C- initial line.

    The velocity expansion is evaluated in the toroidal throat coordinates of
    Kliegel and Levine.  Unlike the fixed-x MOC grid used by the simulator, the
    returned coordinates are the actual characteristic-line coordinates; no
    projection or mass-flow rescaling is applied.
    """

    x_m: np.ndarray
    radius_m: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    axial_velocity_m_s: np.ndarray
    radial_velocity_m_s: np.ndarray
    mach: np.ndarray
    theta_rad: np.ndarray
    discharge_coefficient: float
    subsonic_curvature_radius_m: float


def kliegel_levine_discharge_coefficient(
    gamma: float, subsonic_curvature_ratio: float
) -> float:
    """Return the third-order axisymmetric Kliegel--Levine throat ``Cd``."""
    radius_parameter = 1.0 + float(subsonic_curvature_ratio)
    correction = (gamma + 1.0) / radius_parameter**2 * (
        1.0 / 96.0
        - (8.0 * gamma - 27.0) / (2304.0 * radius_parameter)
        + (754.0 * gamma**2 - 757.0 * gamma + 3633.0)
        / (276480.0 * radius_parameter**2)
    )
    return 1.0 - correction


def _kliegel_levine_velocity_ratios(
    x_over_rt: float | np.ndarray,
    radius_over_rt: float | np.ndarray,
    gamma: float,
    subsonic_curvature_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``u/a*`` and ``v/a*`` from the third-order toroidal series."""
    x = np.asarray(x_over_rt, dtype=float)
    y = np.asarray(radius_over_rt, dtype=float)
    curvature = float(subsonic_curvature_ratio)
    radius_parameter = curvature + 1.0
    z = x * math.sqrt(2.0 * curvature / (gamma + 1.0))

    u1 = 0.5 * y**2 - 0.25 + z
    v1 = 0.25 * y**3 - 0.25 * y + y * z
    u2 = (
        (2.0 * gamma + 9.0) * y**4 / 24.0
        - (4.0 * gamma + 15.0) * y**2 / 24.0
        + (10.0 * gamma + 57.0) / 288.0
        + z * (y**2 - 5.0 / 8.0)
        - (2.0 * gamma - 3.0) * z**2 / 6.0
    )
    v2 = (
        (gamma + 3.0) * y**5 / 9.0
        - (20.0 * gamma + 63.0) * y**3 / 96.0
        + (28.0 * gamma + 93.0) * y / 288.0
        + z
        * ((2.0 * gamma + 9.0) * y**3 / 6.0 - (4.0 * gamma + 15.0) * y / 12.0)
        + y * z**2
    )
    u3 = (
        (556.0 * gamma**2 + 1737.0 * gamma + 3069.0) * y**6 / 10368.0
        - (388.0 * gamma**2 + 1161.0 * gamma + 1881.0) * y**4 / 2304.0
        + (304.0 * gamma**2 + 831.0 * gamma + 1242.0) * y**2 / 1728.0
        - (2708.0 * gamma**2 + 7839.0 * gamma + 14211.0) / 82944.0
        + z
        * (
            (52.0 * gamma**2 + 51.0 * gamma + 327.0) * y**4 / 384.0
            - (52.0 * gamma**2 + 75.0 * gamma + 279.0) * y**2 / 192.0
            + (92.0 * gamma**2 + 180.0 * gamma + 639.0) / 1152.0
        )
        + z**2
        * (-(7.0 * gamma - 3.0) * y**2 / 8.0 + (13.0 * gamma - 27.0) / 48.0)
        + (4.0 * gamma**2 - 57.0 * gamma + 27.0) * z**3 / 144.0
    )
    v3 = (
        (6836.0 * gamma**2 + 23031.0 * gamma + 30627.0) * y**7 / 82944.0
        - (3380.0 * gamma**2 + 11391.0 * gamma + 15291.0) * y**5 / 13824.0
        + (3424.0 * gamma**2 + 11271.0 * gamma + 15228.0) * y**3 / 13824.0
        - (7100.0 * gamma**2 + 22311.0 * gamma + 30249.0) * y / 82944.0
        + z
        * (
            (556.0 * gamma**2 + 1737.0 * gamma + 3069.0) * y**5 / 1728.0
            - (388.0 * gamma**2 + 1161.0 * gamma + 1181.0) * y**3 / 576.0
            + (304.0 * gamma**2 + 831.0 * gamma + 1242.0) * y / 864.0
        )
        + z**2
        * (
            (52.0 * gamma**2 + 51.0 * gamma + 327.0) * y**3 / 192.0
            - (52.0 * gamma**2 + 75.0 * gamma + 279.0) * y / 192.0
        )
        - (7.0 * gamma - 3.0) * y * z**3 / 12.0
    )

    axial_ratio = (
        1.0
        + u1 / radius_parameter
        + (u1 + u2) / radius_parameter**2
        + (u1 + 2.0 * u2 + u3) / radius_parameter**3
    )
    radial_ratio = math.sqrt((gamma + 1.0) / (2.0 * radius_parameter)) * (
        v1 / radius_parameter
        + (1.5 * v1 + v2) / radius_parameter**2
        + (15.0 * v1 / 8.0 + 2.5 * v2 + v3) / radius_parameter**3
    )
    return np.asarray(axial_ratio), np.asarray(radial_ratio)


def build_kliegel_levine_initial_line(
    *,
    stagnation_pressure_pa: float,
    stagnation_temperature_k: float,
    gamma: float,
    gas_constant_j_kg_k: float,
    throat_radius_m: float,
    throat_x_m: float,
    radial_stations: int,
    subsonic_curvature_ratio: float = SAUER_SUBSONIC_CURVATURE_RATIO,
) -> KliegelLevineInitialLine:
    """Construct the genuine curved Kliegel--Levine C- initial-data line.

    Points are placed from the throat wall to the axis.  Each new coordinate
    follows the preceding local ``theta-mu`` characteristic slope.  This is a
    diagnostic initial line for now: the simulator's current fixed-x marcher
    cannot consume it without a characteristic-net bootstrap.
    """
    if radial_stations < 7:
        raise ValueError("Kliegel-Levine radial_stations (Nr) must be at least 7.")
    if not 1.0 < gamma < 2.0:
        raise ValueError("Kliegel-Levine requires a physical gamma between 1 and 2.")
    if min(stagnation_pressure_pa, stagnation_temperature_k, gas_constant_j_kg_k) <= 0.0:
        raise ValueError("Kliegel-Levine requires positive stagnation properties.")
    if throat_radius_m <= 0.0 or subsonic_curvature_ratio <= 0.0:
        raise ValueError("Kliegel-Levine requires positive throat radii.")

    # The 3/2 sine distribution clusters points at both wall and axis, as in
    # established MOC throat-line implementations.
    index = np.arange(radial_stations, dtype=float)
    y_wall_to_axis = np.sin(0.5 * math.pi * (radial_stations - 1 - index) / (radial_stations - 1)) ** 1.5
    x_wall_to_axis = np.zeros(radial_stations)
    mach_wall_to_axis = np.empty(radial_stations)
    theta_wall_to_axis = np.empty(radial_stations)
    q_wall_to_axis = np.empty(radial_stations)
    axial_ratio_wall_to_axis = np.empty(radial_stations)
    radial_ratio_wall_to_axis = np.empty(radial_stations)

    for point in range(radial_stations):
        if point > 0:
            previous_mach = mach_wall_to_axis[point - 1]
            previous_theta = theta_wall_to_axis[point - 1]
            mu = math.asin(1.0 / previous_mach)
            slope = math.tan(previous_theta - mu)
            x_wall_to_axis[point] = x_wall_to_axis[point - 1] + (
                y_wall_to_axis[point] - y_wall_to_axis[point - 1]
            ) / slope

        axial_ratio, radial_ratio = _kliegel_levine_velocity_ratios(
            x_wall_to_axis[point],
            y_wall_to_axis[point],
            gamma,
            subsonic_curvature_ratio,
        )
        axial_ratio = float(axial_ratio)
        radial_ratio = float(radial_ratio)
        q = math.hypot(axial_ratio, radial_ratio)
        sound_speed_ratio_squared = 0.5 * (
            gamma + 1.0 - (gamma - 1.0) * q**2
        )
        if sound_speed_ratio_squared <= 0.0:
            raise ValueError("Kliegel-Levine produced a non-positive local sound speed.")
        mach = q / math.sqrt(sound_speed_ratio_squared)
        if mach <= 1.0:
            raise ValueError(
                "Kliegel-Levine initial line became subsonic; increase the line angle."
            )
        axial_ratio_wall_to_axis[point] = axial_ratio
        radial_ratio_wall_to_axis[point] = radial_ratio
        q_wall_to_axis[point] = q
        mach_wall_to_axis[point] = mach
        theta_wall_to_axis[point] = math.atan2(radial_ratio, axial_ratio)

    # Public arrays use the simulator convention: axis -> wall.
    radius = throat_radius_m * y_wall_to_axis[::-1]
    x = throat_x_m + throat_radius_m * x_wall_to_axis[::-1]
    mach = mach_wall_to_axis[::-1]
    theta = theta_wall_to_axis[::-1]
    critical_speed = math.sqrt(
        2.0 * gamma * gas_constant_j_kg_k * stagnation_temperature_k
        / (gamma + 1.0)
    )
    axial_velocity = critical_speed * axial_ratio_wall_to_axis[::-1]
    radial_velocity = critical_speed * radial_ratio_wall_to_axis[::-1]
    velocity = critical_speed * q_wall_to_axis[::-1]
    temperature = stagnation_temperature_k / (1.0 + 0.5 * (gamma - 1.0) * mach**2)
    pressure = stagnation_pressure_pa * (temperature / stagnation_temperature_k) ** (
        gamma / (gamma - 1.0)
    )
    density = pressure / (gas_constant_j_kg_k * temperature)

    arrays = (x, radius, pressure, temperature, density, velocity, mach, theta)
    if any(not np.isfinite(values).all() for values in arrays):
        raise ArithmeticError("Kliegel-Levine initial line contains non-finite values.")
    return KliegelLevineInitialLine(
        x_m=x,
        radius_m=radius,
        pressure_pa=pressure,
        temperature_k=temperature,
        density_kg_m3=density,
        velocity_m_s=velocity,
        axial_velocity_m_s=axial_velocity,
        radial_velocity_m_s=radial_velocity,
        mach=mach,
        theta_rad=theta,
        discharge_coefficient=kliegel_levine_discharge_coefficient(
            gamma, subsonic_curvature_ratio
        ),
        subsonic_curvature_radius_m=subsonic_curvature_ratio * throat_radius_m,
    )


def curved_line_mass_flow_kg_s(
    line: SauerInitialLine | KliegelLevineInitialLine,
) -> float:
    """Integrate mass flux through an axisymmetric curve ``x=x(r)``."""
    dx_dr = np.gradient(line.x_m, line.radius_m, edge_order=2)
    normal_mass_flux = line.density_kg_m3 * (
        line.axial_velocity_m_s - line.radial_velocity_m_s * dx_dr
    )
    return float(
        2.0 * math.pi * np.trapezoid(normal_mass_flux * line.radius_m, line.radius_m)
    )


def build_sauer_initial_line(
    *,
    stagnation_pressure_pa: float,
    stagnation_temperature_k: float,
    gamma: float,
    gas_constant_j_kg_k: float,
    throat_radius_m: float,
    throat_x_m: float,
    radial_stations: int,
    subsonic_curvature_ratio: float = SAUER_SUBSONIC_CURVATURE_RATIO,
) -> SauerInitialLine:
    """Return the curved, wholly supersonic Sauer initial-data line.

    ``radial_stations`` is :math:`N_r`: the number of points from the
    symmetry axis to the throat wall. Coordinates returned in ``x_m`` are
    absolute simulator coordinates; Sauer's local origin lies at
    ``throat_x_m + eta_m``.
    """
    if radial_stations < 7:
        raise ValueError("Sauer radial_stations (Nr) must be at least 7.")
    if not 1.0 < gamma < 2.0:
        raise ValueError("Sauer requires a physical constant gamma between 1 and 2.")
    if gas_constant_j_kg_k <= 0.0 or stagnation_temperature_k <= 0.0:
        raise ValueError("Sauer requires positive gas constant and stagnation temperature.")
    if stagnation_pressure_pa <= 0.0 or throat_radius_m <= 0.0:
        raise ValueError("Sauer requires positive stagnation pressure and throat radius.")
    if subsonic_curvature_ratio <= 0.0:
        raise ValueError("The subsonic throat-curvature ratio must be positive.")

    rt = float(throat_radius_m)
    rc_sub = float(subsonic_curvature_ratio) * rt
    alpha = math.sqrt(2.0 / ((gamma + 1.0) * rc_sub * rt))
    eta = (gamma + 1.0) * alpha * rt * rt / 8.0
    critical_speed = math.sqrt(
        2.0 * gamma * gas_constant_j_kg_k * stagnation_temperature_k
        / (gamma + 1.0)
    )

    radius = np.linspace(0.0, rt, radial_stations)
    initial_x_relative = 2.0 * eta - 0.25 * (gamma + 1.0) * alpha * radius**2
    local_x = initial_x_relative - eta

    u_perturbation = (
        alpha * local_x + 0.25 * (gamma + 1.0) * alpha**2 * radius**2
    )
    v_perturbation = (
        0.5 * (gamma + 1.0) * alpha**2 * local_x * radius
        + ((gamma + 1.0) ** 2 / 16.0) * alpha**3 * radius**3
    )
    nondimensional_axial_velocity = 1.0 + u_perturbation
    critical_velocity_ratio = np.hypot(
        nondimensional_axial_velocity, v_perturbation
    )
    sound_speed_ratio_squared = (
        0.5 * (gamma + 1.0)
        - 0.5 * (gamma - 1.0) * critical_velocity_ratio**2
    )
    if np.any(sound_speed_ratio_squared <= 0.0):
        raise ValueError(
            "The Sauer expansion produced a non-positive local sound speed; "
            "use a higher-order transonic model for this throat curvature."
        )

    mach = critical_velocity_ratio / np.sqrt(sound_speed_ratio_squared)
    if np.any(mach <= 1.0) or not np.isfinite(mach).all():
        raise ValueError("The Sauer initial-data line is not wholly supersonic.")

    temperature = stagnation_temperature_k / (
        1.0 + 0.5 * (gamma - 1.0) * mach**2
    )
    pressure = stagnation_pressure_pa * (
        temperature / stagnation_temperature_k
    ) ** (gamma / (gamma - 1.0))
    density = pressure / (gas_constant_j_kg_k * temperature)
    axial_velocity = critical_speed * nondimensional_axial_velocity
    radial_velocity = critical_speed * v_perturbation
    velocity = np.hypot(axial_velocity, radial_velocity)
    theta = np.arctan2(radial_velocity, axial_velocity)

    arrays = (
        radius,
        pressure,
        temperature,
        density,
        velocity,
        axial_velocity,
        radial_velocity,
        mach,
        theta,
    )
    if any(not np.isfinite(values).all() for values in arrays):
        raise ArithmeticError("The Sauer initial-data line contains non-finite values.")
    if abs(float(theta[0])) > 1.0e-12 or abs(float(theta[-1])) > 1.0e-10:
        raise ArithmeticError("The Sauer line violates axis or throat-wall tangency.")

    return SauerInitialLine(
        x_m=throat_x_m + initial_x_relative,
        radius_m=radius,
        pressure_pa=pressure,
        temperature_k=temperature,
        density_kg_m3=density,
        velocity_m_s=velocity,
        axial_velocity_m_s=axial_velocity,
        radial_velocity_m_s=radial_velocity,
        mach=mach,
        theta_rad=theta,
        alpha_1_m=float(alpha),
        eta_m=float(eta),
        critical_speed_m_s=float(critical_speed),
        subsonic_curvature_radius_m=float(rc_sub),
    )


def get_initial_transient_line(
    p0: float,
    T0: float,
    gamma: float,
    Rt: float,
    Nr: int,
    gas_constant_j_kg_k: float,
) -> SauerInitialLine:
    """Backward-compatible wrapper for the name used during prototyping."""
    return build_sauer_initial_line(
        stagnation_pressure_pa=p0,
        stagnation_temperature_k=T0,
        gamma=gamma,
        gas_constant_j_kg_k=gas_constant_j_kg_k,
        throat_radius_m=Rt,
        throat_x_m=0.0,
        radial_stations=Nr,
    )
