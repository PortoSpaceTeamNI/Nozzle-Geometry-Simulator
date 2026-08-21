"""Compare fast and converged finalist meshes in the MOC-assisted optimizer."""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

from nozzle_simulator import NozzleInputs
from nozzle_simulator.cea import calculate_ideal_expansion_ratio
from nozzle_simulator.optimization import OptimizationSettings, optimize_geometry


OUTPUT = Path("outputs/moc_optimization_resolution_comparison.csv")
PRESETS = (
    ("fast", 600, 101),
    ("precise", 1200, 201),
)


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
    common = OptimizationSettings(
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
        cache_evaluations=True,
        boundary_layer_model="moc",
        moc_training_samples=24,
        moc_shortlist_size=6,
        moc_refine_candidates=2,
        moc_search_axial_stations=120,
        moc_search_radial_stations=21,
        moc_random_seed=7321,
    )
    rows: list[dict[str, float | int | str]] = []
    for label, axial, radial in PRESETS:
        print(f"Running {label} MOC-assisted optimization ({axial} x {radial})...")
        started = time.perf_counter()
        result = optimize_geometry(
            base,
            replace(
                common,
                moc_refine_axial_stations=axial,
                moc_refine_radial_stations=radial,
            ),
            status=print,
        )
        elapsed = time.perf_counter() - started
        moc = result.moc_result
        if moc is None:
            raise RuntimeError("MOC-assisted optimization returned no exact field.")
        row = {
            "preset": label,
            "axial_stations": axial,
            "radial_stations": radial,
            "expansion_ratio": result.expansion_ratio,
            "bell_fraction": result.bell_fraction,
            "theta_in_deg": result.theta_in_deg,
            "theta_out_deg": result.theta_out_deg,
            "effective_cf": result.fitness,
            "mass_flow_residual_percent": 100.0 * moc.mass_flow_residual,
            "initial_mass_flow_error_percent": 100.0 * moc.initial_mass_flow_error,
            "generations_completed": result.generations_completed,
            "moc_training_evaluations": result.moc_training_evaluations,
            "optimization_time_s": elapsed,
        }
        rows.append(row)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(row, flush=True)
    fast, precise = rows
    print(
        "Fast minus precise: "
        f"dKb={float(fast['bell_fraction']) - float(precise['bell_fraction']):+.6f}, "
        f"dtheta_in={float(fast['theta_in_deg']) - float(precise['theta_in_deg']):+.6f} deg, "
        f"dCf={float(fast['effective_cf']) - float(precise['effective_cf']):+.8f}"
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
