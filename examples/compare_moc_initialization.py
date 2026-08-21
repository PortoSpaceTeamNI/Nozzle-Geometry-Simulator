"""Compare quasi-1D, projected Sauer and Kliegel-Levine MOC initialization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from method_of_caracteristics import (
    MOCSettings,
    analyze_prescribed_nozzle,
    build_kliegel_levine_initial_line,
    build_sauer_initial_line,
    curved_line_mass_flow_kg_s,
)
from nozzle_simulator.models import NozzleInputs
from nozzle_simulator.simulation import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axial", type=int, default=120, help="Axial MOC stations")
    parser.add_argument("--radial", type=int, default=21, help="Radial stations Nr")
    parser.add_argument("--output", type=Path, help="Optional comparison CSV")
    args = parser.parse_args()

    base = simulate(NozzleInputs())
    gamma = float(base.cea.throat.gamma)
    gas_constant = 8.314462618 / (base.cea.throat.molecular_weight_g_mol / 1000.0)
    line_arguments = {
        "stagnation_pressure_pa": base.inputs.chamber_pressure_bar * 1.0e5,
        "stagnation_temperature_k": base.cea.chamber.temperature_k,
        "gamma": gamma,
        "gas_constant_j_kg_k": gas_constant,
        "throat_radius_m": base.inputs.throat_radius_m,
        "throat_x_m": base.geometry.throat_x_m,
        "radial_stations": args.radial,
    }
    sauer_line = build_sauer_initial_line(**line_arguments)
    kliegel_levine_line = build_kliegel_levine_initial_line(**line_arguments)
    cea_mass_flow = (
        base.inputs.chamber_pressure_bar
        * 1.0e5
        * np.pi
        * base.inputs.throat_radius_m**2
        / base.cea.cstar_m_s
    )
    line_rows: list[dict[str, float | str]] = []
    for name, line, discharge_coefficient in (
        ("Sauer", sauer_line, "-"),
        (
            "Kliegel-Levine",
            kliegel_levine_line,
            kliegel_levine_line.discharge_coefficient,
        ),
    ):
        line_mass_flow = curved_line_mass_flow_kg_s(line)
        line_rows.append(
            {
                "curved_line_model": name,
                "radial_stations_Nr": args.radial,
                "mass_flow_kg_s": line_mass_flow,
                "cea_choked_mass_flow_kg_s": cea_mass_flow,
                "error_vs_CEA_percent": 100.0 * (line_mass_flow / cea_mass_flow - 1.0),
                "discharge_coefficient": discharge_coefficient,
                "minimum_Mach": float(np.min(line.mach)),
                "maximum_Mach": float(np.max(line.mach)),
            }
        )
    print(
        ",".join(line_rows[0])
        + "\n"
        + "\n".join(
            ",".join(str(row[key]) for key in row) for row in line_rows
        )
        + "\n"
    )
    rows: list[dict[str, float | str]] = []
    for initialization in ("quasi_1d", "sauer", "kliegel_levine"):
        result = analyze_prescribed_nozzle(
            base.inputs,
            base.geometry,
            base.cea,
            friction_thrust_coefficient=base.performance.friction_thrust_coefficient,
            settings=MOCSettings(
                axial_stations=args.axial,
                radial_stations=args.radial,
                initialization=initialization,
            ),
        )
        rows.append(
            {
                "initialization": initialization,
                "axial_stations": args.axial,
                "radial_stations_Nr": args.radial,
                "initial_M_min": result.start_mach,
                "start_mass_flow_kg_s": float(result.mass_flow_kg_s[0]),
                "exit_mass_flow_kg_s": float(result.mass_flow_kg_s[-1]),
                "cea_choked_mass_flow_kg_s": (
                    base.inputs.chamber_pressure_bar
                    * 1.0e5
                    * np.pi
                    * base.inputs.throat_radius_m**2
                    / base.cea.cstar_m_s
                ),
                "mass_flow_residual_percent": 100.0 * result.mass_flow_residual,
                "inviscid_Cf": result.inviscid_thrust_coefficient,
                "friction_corrected_Cf": result.friction_corrected_thrust_coefficient,
                "exit_M_mean": float(np.mean(result.mach[-1])),
                "exit_M_min": float(np.min(result.mach[-1])),
                "exit_M_max": float(np.max(result.mach[-1])),
                "initialization_time_s": result.initialization_time_s,
                "marching_time_s": result.marching_time_s,
                "total_time_s": result.total_time_s,
            }
        )

    keys = list(rows[0])
    print(",".join(keys))
    for row in rows:
        print(",".join(str(row[key]) for key in keys))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        line_output = args.output.with_name(f"{args.output.stem}_transonic_lines.csv")
        with line_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(line_rows[0]))
            writer.writeheader()
            writer.writerows(line_rows)


if __name__ == "__main__":
    main()
