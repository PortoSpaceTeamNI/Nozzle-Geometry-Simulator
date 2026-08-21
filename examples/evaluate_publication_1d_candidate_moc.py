"""Evaluate the pressure-matched quasi-1D winner with the converged MOC mesh."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from method_of_caracteristics import MOCSettings, analyze_prescribed_nozzle
from nozzle_simulator import NozzleInputs, simulate
from nozzle_simulator.cea import calculate_ideal_expansion_ratio


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "publication_geometry_comparison.csv"
MESH = ROOT / "outputs" / "moc_final_geometry_extended_convergence.csv"
OPTIMIZATION = ROOT / "outputs" / "moc_optimization_resolution_comparison.csv"


def main() -> None:
    with SUMMARY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_case = {row["case"]: row for row in rows}
    base = NozzleInputs(
        chamber_pressure_bar=30.0,
        mixture_ratio=6.5,
        ambient_pressure_atm=0.841,
        throat_radius_m=0.01728,
        chamber_diameter_m=0.096,
        theta_sub_deg=40.0,
        reference_half_angle_deg=15.0,
    )
    base = replace(base, expansion_ratio=calculate_ideal_expansion_ratio(base))
    old = by_case["quasi_1d"]
    old_inputs = replace(
        base,
        bell_fraction=float(old["bell_fraction"]),
        theta_in_deg=float(old["theta_in_deg"]),
    )
    old_simulation = simulate(old_inputs)
    evaluation = analyze_prescribed_nozzle(
        old_simulation.inputs,
        old_simulation.geometry,
        old_simulation.cea,
        friction_thrust_coefficient=(
            old_simulation.performance.friction_thrust_coefficient
        ),
        settings=MOCSettings(
            axial_stations=1200,
            radial_stations=201,
            initialization="kliegel_levine",
        ),
    )

    with MESH.open(newline="", encoding="utf-8") as stream:
        initial = next(
            row for row in csv.DictReader(stream) if row["resolution"] == "1200x201"
        )
    with OPTIMIZATION.open(newline="", encoding="utf-8") as stream:
        final = next(row for row in csv.DictReader(stream) if row["preset"] == "precise")

    precise_values = {
        "initial": (
            initial["friction_corrected_cf"],
            initial["mass_flow_residual_percent"],
            initial["initial_mass_flow_error_percent"],
        ),
        "quasi_1d": (
            evaluation.friction_corrected_thrust_coefficient,
            100.0 * evaluation.mass_flow_residual,
            100.0 * evaluation.initial_mass_flow_error,
        ),
        "moc_optimized": (
            final["effective_cf"],
            final["mass_flow_residual_percent"],
            final["initial_mass_flow_error_percent"],
        ),
    }
    for row in rows:
        cf, residual, initial_error = precise_values[row["case"]]
        row["precise_moc_cf"] = cf
        row["moc_mass_flow_residual_percent"] = residual
        row["moc_initial_mass_flow_error_percent"] = initial_error
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Updated {SUMMARY}")


if __name__ == "__main__":
    main()
