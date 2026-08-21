"""Export simulation inputs and profiles to a self-contained result folder."""

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import SimulationResult
from .performance import loss_breakdown


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def export_result(result: SimulationResult, parent: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(parent) / f"nozzle_run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "metadata": result.metadata,
        "inputs": asdict(result.inputs),
        "cea": asdict(result.cea),
        "geometry": {
            "throat_x_m": result.geometry.throat_x_m,
            "exit_x_m": result.geometry.exit_x_m,
            "exit_radius_m": result.geometry.exit_radius_m,
            "divergent_length_m": result.geometry.divergent_length_m,
            "total_length_m": result.geometry.total_length_m,
            "cone_length_m": result.geometry.cone_length_m,
            "theta_out_deg": result.geometry.theta_out_deg,
            "contraction_ratio": result.geometry.contraction_ratio,
            "bell_fraction": result.inputs.bell_fraction,
            "bell_parabola_coefficients": result.geometry.bell_coefficients,
        },
        "boundary_layer": {
            "velocity_efficiency": result.boundary_layer.velocity_efficiency,
        },
        "performance": asdict(result.performance),
        "loss_breakdown_cf": loss_breakdown(result.performance),
    }
    if result.moc is not None:
        moc = result.moc
        summary["axisymmetric_moc"] = {
            "converged": moc.converged,
            "initialization": moc.initialization,
            "start_mach": moc.start_mach,
            "initial_line_mach_min": (
                float(np.min(moc.initial_line_mach))
                if moc.initial_line_mach.size
                else None
            ),
            "radial_stations_Nr": int(moc.radial_fraction.size),
            "transition_characteristic_lines": len(moc.transition_line_x_m),
            "gamma": moc.gamma,
            "gas_constant_j_kg_k": moc.gas_constant_j_kg_k,
            "sauer_alpha_1_m": (
                moc.sauer_alpha_1_m if np.isfinite(moc.sauer_alpha_1_m) else None
            ),
            "sauer_eta_m": moc.sauer_eta_m if np.isfinite(moc.sauer_eta_m) else None,
            "initialization_time_s": moc.initialization_time_s,
            "marching_time_s": moc.marching_time_s,
            "total_time_s": moc.total_time_s,
            "cea_choked_mass_flow_kg_s": moc.cea_choked_mass_flow_kg_s,
            "initial_mass_flow_error": moc.initial_mass_flow_error,
            "mass_flow_residual": moc.mass_flow_residual,
            "inviscid_thrust_n": moc.inviscid_thrust_n,
            "inviscid_thrust_coefficient": moc.inviscid_thrust_coefficient,
            "friction_corrected_thrust_coefficient": (
                moc.friction_corrected_thrust_coefficient
            ),
            "warnings": list(moc.warnings),
        }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    g, f, t, bl = result.geometry, result.flow, result.thermal, result.boundary_layer
    _write_csv(
        output / "geometry.csv",
        ["x_mm", "radius_mm", "z_mm"],
        zip(g.x_m * 1e3, g.radius_m * 1e3, [0.0] * len(g.x_m)),
    )
    _write_csv(
        output / "flow_profile.csv",
        ["x_m", "radius_m", "mach", "temperature_K", "pressure_bar", "gamma"],
        zip(g.x_m, g.radius_m, f.mach, f.temperature_k, f.pressure_bar, f.gamma),
    )
    _write_csv(
        output / "thermal_profile.csv",
        ["x_m", "Taw_K", "Tw_K", "hg_W_m2K", "heat_flux_W_m2"],
        zip(
            g.x_m,
            t.adiabatic_wall_temperature_k,
            t.wall_temperature_k,
            t.heat_transfer_coefficient_w_m2_k,
            t.heat_flux_w_m2,
        ),
    )
    _write_csv(
        output / "boundary_layer.csv",
        [
            "x_m",
            "delta_star_m",
            "momentum_thickness_m",
            "shape_factor",
            "skin_friction_coefficient",
            "wall_shear_stress_Pa",
            "wall_temperature_K",
            "effective_radius_m",
            "Re_s",
            "mu_edge_Pa_s",
            "Mach_effective",
        ],
        zip(
            g.x_m,
            bl.displacement_thickness_m,
            bl.momentum_thickness_m,
            bl.shape_factor,
            bl.skin_friction_coefficient,
            bl.wall_shear_stress_pa,
            bl.wall_temperature_k,
            bl.effective_radius_m,
            bl.reynolds,
            bl.viscosity_pa_s,
            bl.mach_effective,
        ),
    )
    if result.moc is not None:
        moc = result.moc
        if moc.initial_line_x_m.size:
            _write_csv(
                output / "moc_initial_data_line.csv",
                [
                    "x_m",
                    "radius_m",
                    "Mach",
                    "theta_deg",
                    "pressure_bar",
                ],
                zip(
                    moc.initial_line_x_m,
                    moc.initial_line_radius_m,
                    moc.initial_line_mach,
                    np.degrees(moc.initial_line_theta_rad),
                    moc.initial_line_pressure_pa / 1.0e5,
                ),
            )
        if moc.transition_line_x_m:
            transition_rows = []
            for line_index, (line_x, line_radius) in enumerate(
                zip(moc.transition_line_x_m, moc.transition_line_radius_m)
            ):
                transition_rows.extend(
                    (line_index, point_index, x_value, radius_value)
                    for point_index, (x_value, radius_value) in enumerate(
                        zip(line_x, line_radius)
                    )
                )
            _write_csv(
                output / "moc_transition_net.csv",
                ["characteristic_line", "point", "x_m", "radius_m"],
                transition_rows,
            )
        _write_csv(
            output / "moc_station_diagnostics.csv",
            ["x_m", "wall_radius_m", "mass_flow_kg_s"],
            zip(moc.x_m, moc.radius_m[:, -1], moc.mass_flow_kg_s),
        )
        _write_csv(
            output / "moc_exit_profile.csv",
            [
                "radius_m",
                "radius_fraction",
                "pressure_bar",
                "temperature_K",
                "density_kg_m3",
                "velocity_m_s",
                "axial_velocity_m_s",
                "Mach",
                "theta_deg",
            ],
            zip(
                moc.radius_m[-1],
                moc.radial_fraction,
                moc.pressure_pa[-1] / 1.0e5,
                moc.temperature_k[-1],
                moc.density_kg_m3[-1],
                moc.velocity_m_s[-1],
                moc.axial_velocity_m_s[-1],
                moc.mach[-1],
                moc.exit_theta_deg,
            ),
        )
        _write_csv(
            output / "moc_field.csv",
            [
                "x_m",
                "radius_m",
                "radius_fraction",
                "pressure_bar",
                "temperature_K",
                "density_kg_m3",
                "velocity_m_s",
                "axial_velocity_m_s",
                "Mach",
                "theta_deg",
            ],
            (
                (
                    moc.x_m[i],
                    moc.radius_m[i, j],
                    moc.radial_fraction[j],
                    moc.pressure_pa[i, j] / 1.0e5,
                    moc.temperature_k[i, j],
                    moc.density_kg_m3[i, j],
                    moc.velocity_m_s[i, j],
                    moc.axial_velocity_m_s[i, j],
                    moc.mach[i, j],
                    np.degrees(moc.theta_rad[i, j]),
                )
                for i in range(moc.x_m.size)
                for j in range(moc.radial_fraction.size)
            ),
        )
    return output
