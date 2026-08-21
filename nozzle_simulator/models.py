"""Typed data exchanged by the simulator modules."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ATM_TO_BAR = 1.01325


@dataclass(frozen=True)
class NozzleInputs:
    chamber_pressure_bar: float = 30.0
    mixture_ratio: float = 6.5
    ambient_pressure_atm: float = 1.0
    expansion_ratio: float = 5.6
    throat_radius_m: float = 0.01728
    chamber_diameter_m: float = 0.120
    reference_half_angle_deg: float = 15.0
    bell_fraction: float = 0.80
    theta_in_deg: float = 30.0
    theta_sub_deg: float = 50.0

    @property
    def ambient_pressure_bar(self) -> float:
        return self.ambient_pressure_atm * ATM_TO_BAR

    def validate(self) -> None:
        positive = {
            "Chamber pressure": self.chamber_pressure_bar,
            "O/F": self.mixture_ratio,
            "Expansion ratio": self.expansion_ratio,
            "Throat radius": self.throat_radius_m,
            "Chamber diameter": self.chamber_diameter_m,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.ambient_pressure_atm < 0.0:
            raise ValueError("Ambient pressure cannot be negative.")
        if self.expansion_ratio <= 1.0:
            raise ValueError("Expansion ratio must be greater than 1.")
        if not 0.0 < self.bell_fraction <= 1.5:
            raise ValueError("Bell fraction must be between 0 and 150%.")
        if not 0.0 < self.theta_in_deg < 90.0:
            raise ValueError("Initial wall angle must be between 0 and 90 degrees.")
        if not 0.0 < self.theta_sub_deg < 90.0:
            raise ValueError("Convergent angle must be between 0 and 90 degrees.")
        if not 0.0 < self.reference_half_angle_deg < 90.0:
            raise ValueError("Reference cone half-angle must be between 0 and 90 degrees.")


@dataclass(frozen=True)
class StationProperties:
    temperature_k: float
    molecular_weight_g_mol: float
    gamma: float
    cp_j_kg_k: float
    viscosity_pa_s: float
    conductivity_w_m_k: float
    prandtl: float


@dataclass(frozen=True)
class CEAProperties:
    chamber: StationProperties
    throat: StationProperties
    exit: StationProperties
    cstar_m_s: float
    exit_mach: float
    chamber_to_exit_pressure_ratio: float
    ideal_momentum_thrust_coefficient: float
    ambient_thrust_coefficient: float
    ambient_mode: str
    ideal_expansion_ratio: float | None

    @property
    def exit_pressure_bar(self) -> float:
        return self.chamber_to_exit_pressure_ratio and (
            self._chamber_pressure_bar / self.chamber_to_exit_pressure_ratio
        )

    _chamber_pressure_bar: float = field(repr=False, default=0.0)


@dataclass
class GeometryResult:
    x_m: np.ndarray
    radius_m: np.ndarray
    segments: dict[str, tuple[np.ndarray, np.ndarray]]
    throat_x_m: float
    exit_x_m: float
    exit_radius_m: float
    divergent_length_m: float
    total_length_m: float
    cone_length_m: float
    theta_out_deg: float
    contraction_ratio: float
    bell_coefficients: tuple[float, float, float]

    @property
    def parabola_coefficients(self) -> tuple[float, float, float]:
        """Alias retaining the established name for the quadratic bell coefficients."""
        return self.bell_coefficients


@dataclass
class FlowResult:
    mach: np.ndarray
    temperature_k: np.ndarray
    pressure_bar: np.ndarray
    gamma: np.ndarray


@dataclass
class ThermalResult:
    adiabatic_wall_temperature_k: np.ndarray
    wall_temperature_k: np.ndarray
    heat_transfer_coefficient_w_m2_k: np.ndarray
    heat_flux_w_m2: np.ndarray


@dataclass
class BoundaryLayerResult:
    displacement_thickness_m: np.ndarray
    momentum_thickness_m: np.ndarray
    shape_factor: np.ndarray
    skin_friction_coefficient: np.ndarray
    wall_shear_stress_pa: np.ndarray
    wall_temperature_k: np.ndarray
    effective_radius_m: np.ndarray
    reynolds: np.ndarray
    viscosity_pa_s: np.ndarray
    mach_effective: np.ndarray
    velocity_efficiency: float


@dataclass(frozen=True)
class PerformanceResult:
    divergence_efficiency: float
    momentum_efficiency: float
    momentum_thrust_coefficient: float
    pressure_thrust_coefficient: float
    friction_thrust_coefficient: float
    effective_thrust_coefficient: float
    effective_thrust_n: float


@dataclass
class MOCResult:
    """Axisymmetric prescribed-wall Method-of-Characteristics solution."""

    x_m: np.ndarray
    radial_fraction: np.ndarray
    radius_m: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    mach: np.ndarray
    theta_rad: np.ndarray
    axial_velocity_m_s: np.ndarray
    mass_flow_kg_s: np.ndarray
    mass_flow_residual: float
    inviscid_thrust_n: float
    inviscid_thrust_coefficient: float
    friction_corrected_thrust_coefficient: float
    start_mach: float
    gamma: float
    gas_constant_j_kg_k: float
    converged: bool
    initialization: str
    initial_line_x_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    initial_line_radius_m: np.ndarray = field(default_factory=lambda: np.empty(0))
    initial_line_mach: np.ndarray = field(default_factory=lambda: np.empty(0))
    initial_line_theta_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    initial_line_pressure_pa: np.ndarray = field(default_factory=lambda: np.empty(0))
    transition_line_x_m: tuple[np.ndarray, ...] = ()
    transition_line_radius_m: tuple[np.ndarray, ...] = ()
    sauer_alpha_1_m: float = float("nan")
    sauer_eta_m: float = float("nan")
    initialization_time_s: float = 0.0
    marching_time_s: float = 0.0
    total_time_s: float = 0.0
    cea_choked_mass_flow_kg_s: float = float("nan")
    initial_mass_flow_error: float = float("nan")
    warnings: tuple[str, ...] = ()

    @property
    def exit_pressure_bar(self) -> np.ndarray:
        return self.pressure_pa[-1] / 1.0e5

    @property
    def exit_theta_deg(self) -> np.ndarray:
        return np.degrees(self.theta_rad[-1])


@dataclass
class SimulationResult:
    inputs: NozzleInputs
    cea: CEAProperties
    geometry: GeometryResult
    flow: FlowResult
    thermal: ThermalResult
    boundary_layer: BoundaryLayerResult
    performance: PerformanceResult
    moc: MOCResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
