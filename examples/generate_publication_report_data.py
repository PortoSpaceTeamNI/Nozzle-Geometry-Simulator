"""Generate reproducible data files used by the publication report."""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

from nozzle_simulator import NozzleInputs, simulate
from nozzle_simulator.cea import calculate_ideal_expansion_ratio
from nozzle_simulator.geometry import build_geometry
from nozzle_simulator.optimization import OptimizationSettings, optimize_geometry


ROOT = Path(__file__).resolve().parents[1]
REPORT_IMAGES = ROOT / "report_final_publication" / "Images"
SUMMARY = ROOT / "outputs" / "publication_geometry_comparison.csv"
MOC_COMPARISON = ROOT / "outputs" / "moc_optimization_resolution_comparison.csv"


def _write_geometry(name: str, inputs: NozzleInputs) -> None:
    geometry = build_geometry(inputs)
    path = REPORT_IMAGES / f"{name}_geometry_final.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x_mm", "radius_mm"))
        writer.writerows(zip(1.0e3 * geometry.x_m, 1.0e3 * geometry.radius_m))


def main() -> None:
    base = NozzleInputs(
        chamber_pressure_bar=30.0,
        mixture_ratio=6.5,
        ambient_pressure_atm=0.841,
        throat_radius_m=0.01728,
        chamber_diameter_m=0.096,
        bell_fraction=0.80,
        theta_in_deg=30.0,
        theta_sub_deg=40.0,
        reference_half_angle_deg=15.0,
    )
    base = replace(base, expansion_ratio=calculate_ideal_expansion_ratio(base))
    configuration = OptimizationSettings(
        bell_fraction_min=0.60,
        bell_fraction_max=1.00,
        theta_in_min_deg=20.0,
        theta_in_max_deg=45.0,
        num_generations=300,
        population_size=100,
        num_parents_mating=20,
        keep_elitism=3,
        saturation_generations=40,
        crossover_probability=0.85,
        mutation_percent_high=67,
        mutation_percent_low=34,
        evaluation_mode="processes",
        parallel_workers=4,
        boundary_layer_model="blimp",
        moc_random_seed=7321,
    )
    print("Running pressure-matched quasi-1D/BLIMP reference optimization...")
    started = time.perf_counter()
    old_1d = optimize_geometry(base, configuration)
    old_1d_time = time.perf_counter() - started
    old_1d_inputs = replace(
        base,
        bell_fraction=old_1d.bell_fraction,
        theta_in_deg=old_1d.theta_in_deg,
    )

    with MOC_COMPARISON.open(newline="", encoding="utf-8") as stream:
        precise = next(
            row for row in csv.DictReader(stream) if row["preset"] == "precise"
        )
    moc_inputs = replace(
        base,
        bell_fraction=float(precise["bell_fraction"]),
        theta_in_deg=float(precise["theta_in_deg"]),
    )

    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)
    for name, inputs in (
        ("initial", base),
        ("quasi_1d", old_1d_inputs),
        ("moc_optimized", moc_inputs),
    ):
        _write_geometry(name, inputs)

    rows: list[dict[str, float | int | str]] = []
    for name, inputs, reported_cf, elapsed in (
        ("initial", base, "", 0.0),
        ("quasi_1d", old_1d_inputs, old_1d.fitness, old_1d_time),
        ("moc_optimized", moc_inputs, float(precise["effective_cf"]), float(precise["optimization_time_s"])),
    ):
        simulation = simulate(inputs)
        rows.append(
            {
                "case": name,
                "expansion_ratio": inputs.expansion_ratio,
                "bell_fraction": inputs.bell_fraction,
                "theta_in_deg": inputs.theta_in_deg,
                "theta_out_deg": simulation.geometry.theta_out_deg,
                "divergent_length_mm": 1.0e3 * simulation.geometry.divergent_length_m,
                "total_length_mm": 1.0e3 * simulation.geometry.total_length_m,
                "quasi_1d_blimp_cf": simulation.performance.effective_thrust_coefficient,
                "reported_optimization_cf": reported_cf,
                "optimization_time_s": elapsed,
                "generations_completed": (
                    old_1d.generations_completed
                    if name == "quasi_1d"
                    else int(precise["generations_completed"])
                    if name == "moc_optimized"
                    else 0
                ),
            }
        )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved {SUMMARY} and geometry CSV files in {REPORT_IMAGES}")


if __name__ == "__main__":
    main()
