"""Compare the legacy quasi-1D objective with MOC-assisted optimization."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from nozzle_simulator import NozzleInputs
from nozzle_simulator.cea import calculate_ideal_expansion_ratio
from nozzle_simulator.optimization import (
    OptimizationSettings,
    evaluate_contour,
    evaluate_moc_contour,
    optimize_geometry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--moc-samples", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/optimization_fixed_epsilon_1d_vs_moc.csv"),
    )
    return parser


def _solution(result) -> tuple[float, float]:
    return result.bell_fraction, result.theta_in_deg


def main() -> None:
    args = _parser().parse_args()
    base = NozzleInputs(
        chamber_pressure_bar=30.0,
        mixture_ratio=6.5,
        ambient_pressure_atm=0.841,
        throat_radius_m=0.01728,
        chamber_diameter_m=0.120,
        theta_sub_deg=50.0,
    )
    base = replace(base, expansion_ratio=calculate_ideal_expansion_ratio(base))
    common = OptimizationSettings(
        bell_fraction_min=0.55,
        bell_fraction_max=1.00,
        theta_in_min_deg=20.0,
        theta_in_max_deg=45.0,
        num_generations=args.generations,
        population_size=args.population,
        num_parents_mating=min(12, args.population),
        keep_elitism=min(3, args.population - 1),
        saturation_generations=min(25, args.generations),
        evaluation_mode="processes",
        parallel_workers=args.workers,
        moc_training_samples=args.moc_samples,
        moc_shortlist_size=8,
        moc_refine_candidates=3,
        moc_search_axial_stations=120,
        moc_search_radial_stations=21,
        moc_refine_axial_stations=360,
        moc_refine_radial_stations=61,
        moc_random_seed=7321,
    )

    print("Running quasi-1D + BLIMP optimization...")
    started = time.perf_counter()
    one_d = optimize_geometry(base, replace(common, boundary_layer_model="blimp"))
    one_d_elapsed = time.perf_counter() - started

    print("Running MOC-assisted optimization...")
    started = time.perf_counter()
    moc = optimize_geometry(
        base,
        replace(common, boundary_layer_model="moc"),
        status=print,
    )
    moc_elapsed = time.perf_counter() - started

    print("Cross-evaluating both winners at the refined MOC resolution...")
    one_d_moc = evaluate_moc_contour(
        base,
        _solution(one_d),
        axial_stations=common.moc_refine_axial_stations,
        radial_stations=common.moc_refine_radial_stations,
    )
    moc_exact = moc.moc_result
    rows = [
        {
            "optimizer": "quasi_1d_blimp",
            "expansion_ratio": one_d.expansion_ratio,
            "bell_fraction": one_d.bell_fraction,
            "theta_in_deg": one_d.theta_in_deg,
            "theta_out_deg": one_d.theta_out_deg,
            "exit_pressure_bar": one_d.exit_pressure_bar,
            "quasi_1d_blimp_cf": evaluate_contour(base, _solution(one_d), "blimp"),
            "refined_moc_cf": one_d_moc.fitness,
            "moc_mass_residual": one_d_moc.mass_flow_residual,
            "search_seconds": one_d_elapsed,
        },
        {
            "optimizer": "moc_assisted",
            "expansion_ratio": moc.expansion_ratio,
            "bell_fraction": moc.bell_fraction,
            "theta_in_deg": moc.theta_in_deg,
            "theta_out_deg": moc.theta_out_deg,
            "exit_pressure_bar": moc.exit_pressure_bar,
            "quasi_1d_blimp_cf": evaluate_contour(base, _solution(moc), "blimp"),
            "refined_moc_cf": moc.fitness,
            "moc_mass_residual": moc_exact.mass_flow_residual,
            "search_seconds": moc_elapsed,
        },
    ]
    legacy_path = Path("outputs/optimization_1d_vs_moc.csv")
    if legacy_path.exists():
        with legacy_path.open(newline="", encoding="utf-8") as stream:
            legacy = next(
                (
                    row
                    for row in csv.DictReader(stream)
                    if row["optimizer"] == "quasi_1d_blimp"
                ),
                None,
            )
        if legacy is not None:
            legacy["optimizer"] = "legacy_three_gene_1d"
            rows.insert(0, legacy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
