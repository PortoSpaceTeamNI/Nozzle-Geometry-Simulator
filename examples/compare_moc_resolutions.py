"""Run a MOC mesh-convergence study for the final pressure-matched contour."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from method_of_caracteristics import MOCSettings, analyze_prescribed_nozzle
from nozzle_simulator import NozzleInputs, simulate
from nozzle_simulator.cea import calculate_ideal_expansion_ratio

RESOLUTIONS = (
    (140, 41),
    (240, 41),
    (360, 61),
    (480, 81),
    (600, 101),
    (720, 121),
    (840, 141),
    (960, 161),
    (1080, 181),
    (1200, 201),
    (1320, 221),
    (1440, 241),
)
CF_CONVERGENCE_PERCENT = 0.1
MASS_RESIDUAL_TARGET_PERCENT = 0.5
INITIAL_MASS_FLOW_TARGET_PERCENT = 0.5


def _save_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    serializable: list[dict[str, float | int | str]] = []
    finest_cf = float(rows[-1]["friction_corrected_cf"])
    stable_streak = 0
    for index, source in enumerate(rows):
        row = dict(source)
        current_cf = float(row["friction_corrected_cf"])
        if index:
            previous_cf = float(rows[index - 1]["friction_corrected_cf"])
            step_delta = current_cf - previous_cf
            row["delta_cf_from_previous"] = step_delta
            step_change = 100.0 * abs(step_delta) / current_cf
            row["step_cf_change_percent"] = step_change
            stable_streak = stable_streak + 1 if step_change < CF_CONVERGENCE_PERCENT else 0
        else:
            row["delta_cf_from_previous"] = ""
            row["step_cf_change_percent"] = ""
        residual_ok = (
            float(row["mass_flow_residual_percent"])
            < MASS_RESIDUAL_TARGET_PERCENT
        )
        initial_flow_ok = (
            float(row["initial_mass_flow_error_percent"])
            < INITIAL_MASS_FLOW_TARGET_PERCENT
        )
        row["mass_flow_residual_acceptable"] = residual_ok
        row["initial_mass_flow_acceptable"] = initial_flow_ok
        row["cf_convergence_streak"] = stable_streak
        row["mesh_converged"] = bool(
            int(row["axial_stations"]) >= 600
            and stable_streak >= 2
            and residual_ok
            and initial_flow_ok
        )
        delta = current_cf - finest_cf
        row["delta_cf_vs_finest"] = delta
        row["relative_cf_error_vs_finest_percent"] = 100.0 * delta / finest_cf
        serializable.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in serializable:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/moc_final_geometry_extended_convergence.csv"),
    )
    args = parser.parse_args()

    inputs = NozzleInputs(
        chamber_pressure_bar=30.0,
        mixture_ratio=6.5,
        ambient_pressure_atm=0.841,
        throat_radius_m=0.01728,
        chamber_diameter_m=0.096,
        bell_fraction=0.80260266,
        theta_in_deg=32.351992,
        theta_sub_deg=40.0,
        reference_half_angle_deg=15.0,
    )
    inputs = replace(
        inputs,
        expansion_ratio=calculate_ideal_expansion_ratio(inputs),
    )
    baseline = simulate(inputs)
    cea_choked_mass_flow = (
        inputs.chamber_pressure_bar
        * 1.0e5
        * math.pi
        * inputs.throat_radius_m**2
        / baseline.cea.cstar_m_s
    )
    rows: list[dict[str, float | int | str]] = []
    seed_path = (
        args.output
        if args.output.exists()
        else Path("outputs/moc_final_geometry_mesh_convergence.csv")
    )
    if seed_path.exists():
        allowed = {f"{axial}x{radial}" for axial, radial in RESOLUTIONS}
        with seed_path.open(newline="", encoding="utf-8") as stream:
            for source in csv.DictReader(stream):
                if source["resolution"] not in allowed:
                    continue
                start_mass_flow = float(source["start_mass_flow_kg_s"])
                initial_error_percent = (
                    100.0
                    * abs(start_mass_flow - cea_choked_mass_flow)
                    / cea_choked_mass_flow
                )
                if initial_error_percent > INITIAL_MASS_FLOW_TARGET_PERCENT:
                    print(
                        f"Discarding saved {source['resolution']}: initial mass-flow "
                        f"error {initial_error_percent:.3f}% exceeds "
                        f"{INITIAL_MASS_FLOW_TARGET_PERCENT}%.",
                        flush=True,
                    )
                    continue
                rows.append(
                    {
                        key: value
                        for key, value in source.items()
                        if key
                        not in {
                            "delta_cf_vs_480x81",
                            "relative_cf_error_percent",
                            "delta_cf_from_previous",
                            "step_cf_change_percent",
                            "delta_cf_vs_finest",
                            "relative_cf_error_vs_finest_percent",
                        }
                    }
                )
                rows[-1]["cea_choked_mass_flow_kg_s"] = cea_choked_mass_flow
                rows[-1]["initial_mass_flow_error_percent"] = initial_error_percent
        print(f"Resuming from {seed_path} with {len(rows)} saved meshes.", flush=True)

    completed = {str(row["resolution"]) for row in rows}
    stable_steps = 0
    for axial, radial in RESOLUTIONS:
        resolution = f"{axial}x{radial}"
        if resolution in completed:
            continue
        print(f"Running Kliegel-Levine MOC {axial} x {radial}...", flush=True)
        wall_started = time.perf_counter()
        result = analyze_prescribed_nozzle(
            baseline.inputs,
            baseline.geometry,
            baseline.cea,
            friction_thrust_coefficient=(
                baseline.performance.friction_thrust_coefficient
            ),
            settings=MOCSettings(
                axial_stations=axial,
                radial_stations=radial,
                initialization="kliegel_levine",
            ),
        )
        wall_time = time.perf_counter() - wall_started
        rows.append(
            {
                "resolution": resolution,
                "axial_stations": axial,
                "radial_stations": radial,
                "expansion_ratio": inputs.expansion_ratio,
                "bell_fraction": inputs.bell_fraction,
                "theta_in_deg": inputs.theta_in_deg,
                "theta_out_deg": baseline.geometry.theta_out_deg,
                "mass_flow_residual_percent": 100.0 * result.mass_flow_residual,
                "start_mass_flow_kg_s": float(result.mass_flow_kg_s[0]),
                "exit_mass_flow_kg_s": float(result.mass_flow_kg_s[-1]),
                "cea_choked_mass_flow_kg_s": result.cea_choked_mass_flow_kg_s,
                "initial_mass_flow_error_percent": (
                    100.0 * result.initial_mass_flow_error
                ),
                "inviscid_cf": result.inviscid_thrust_coefficient,
                "friction_corrected_cf": (
                    result.friction_corrected_thrust_coefficient
                ),
                "mean_exit_pressure_bar": float(
                    np.mean(result.exit_pressure_bar)
                ),
                "maximum_exit_angle_deg": float(
                    np.max(np.abs(result.exit_theta_deg))
                ),
                "total_time_s": result.total_time_s,
                "wall_time_s": wall_time,
            }
        )
        print(rows[-1], flush=True)
        _save_rows(rows, args.output)
        if len(rows) >= 2:
            current_cf = float(rows[-1]["friction_corrected_cf"])
            previous_cf = float(rows[-2]["friction_corrected_cf"])
            change_percent = 100.0 * abs(current_cf - previous_cf) / current_cf
            stable_steps = stable_steps + 1 if change_percent < CF_CONVERGENCE_PERCENT else 0
            residual_percent = float(rows[-1]["mass_flow_residual_percent"])
            initial_error_percent = float(
                rows[-1]["initial_mass_flow_error_percent"]
            )
            if (
                axial >= 600
                and stable_steps >= 2
                and residual_percent < MASS_RESIDUAL_TARGET_PERCENT
                and initial_error_percent < INITIAL_MASS_FLOW_TARGET_PERCENT
            ):
                print(
                    "Convergence reached: two successive Cf changes below "
                    f"{CF_CONVERGENCE_PERCENT}% and mass residual below "
                    f"{MASS_RESIDUAL_TARGET_PERCENT}%.",
                    flush=True,
                )
                break

    _save_rows(rows, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
