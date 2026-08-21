"""GA para otimizar apenas o contorno de um bell nozzle pré-dimensionado."""

import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygad
import yaml

try:
    from .specific_impulse_objective import (
        EPS_FIXED, ISP_REF, P0_OPER, P_AMB, R_CHAMBER_FIXED, R_T_FIXED,
        T0_OPER, W_OPER, evaluate_solution, fitness_function,
    )
    from .reduced_order_nozzle_performance import (
        compute_convergent_geometry, compute_parabola_coefficients,
    )
except ImportError:
    from specific_impulse_objective import (
        EPS_FIXED, ISP_REF, P0_OPER, P_AMB, R_CHAMBER_FIXED, R_T_FIXED,
        T0_OPER, W_OPER, evaluate_solution, fitness_function,
    )
    from reduced_order_nozzle_performance import (
        compute_convergent_geometry, compute_parabola_coefficients,
    )


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "Resultados")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TOP_N = 5

history = {"best": [], "mean": [], "std": [], "solution": []}


def on_generation(ga):
    fits = np.asarray(ga.last_generation_fitness, dtype=float)
    idx = int(np.argmax(fits))
    history["best"].append(float(fits[idx]))
    history["mean"].append(float(np.mean(fits)))
    history["std"].append(float(np.std(fits)))
    history["solution"].append(ga.population[idx].copy())
    if ga.generations_completed == 1 or ga.generations_completed % 25 == 0:
        print(f"Gen {ga.generations_completed:4d} | best={fits[idx]:.6f} | "
              f"mean={np.mean(fits):.6f}")


def build_ga(num_generations=300, sol_per_pop=100):
    """Cria o GA. Os três genes são [% bell, theta_in, theta_sub]."""
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=min(20, sol_per_pop),
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        gene_space=[
            {"low": 0.60, "high": 1.00},  # L_div / L_cone,15deg
            {"low": 20.0, "high": 35.0}, # theta_in [deg]
            {"low": 40.0, "high": 65.0}, # theta_sub [deg]
        ],
        gene_type=float,
        num_genes=3,
        mutation_percent_genes=[67, 34],
        parent_selection_type="tournament",
        crossover_type="uniform",
        crossover_probability=0.85,
        mutation_type="adaptive",
        keep_elitism=3,
        stop_criteria="saturate_40",
        save_solutions=False,
        on_generation=on_generation,
    )


def nozzle_contour(solution, n=400):
    bell_fraction, theta_in, theta_sub = solution
    x_conv, r_conv, _ = compute_convergent_geometry(
        R_T_FIXED, theta_sub, R_CHAMBER_FIXED)
    a, b, c, x_p, length = compute_parabola_coefficients(
        R_T_FIXED, EPS_FIXED, theta_in, bell_fraction)
    x_div = np.linspace(x_p, length, n)
    r_div = a * x_div**2 + b * x_div + c
    return np.concatenate((x_conv, x_div)) * 1e3, np.concatenate((r_conv, r_div)) * 1e3


def save_plots(ga):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    generations = np.arange(1, len(history["best"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(generations, history["best"], label="best")
    axes[0].plot(generations, history["mean"], label="mean")
    axes[0].set(xlabel="Generation", ylabel="Isp / Isp_ref", title="Fitness")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    solutions = np.asarray(history["solution"])
    for index, label in enumerate(("Bell fraction", "theta_in", "theta_sub")):
        axes[1].plot(generations, solutions[:, index], label=label)
    axes[1].set(xlabel="Generation", title="Best chromosome")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fits = np.asarray(ga.last_generation_fitness)
    for rank, idx in enumerate(np.argsort(fits)[::-1][:TOP_N], start=1):
        try:
            x, r = nozzle_contour(ga.population[idx])
            axes[2].plot(x, r, label=f"#{rank}: {fits[idx]:.5f}")
        except ValueError:
            pass
    axes[2].set(xlabel="x [mm]", ylabel="r [mm]", title="Best contours")
    axes[2].axis("equal"); axes[2].grid(alpha=0.3); axes[2].legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"postprocessing_isp_{STAMP}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_yaml(solution, result, fitness):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bell_fraction, theta_in, theta_sub = map(float, solution)
    data = {
        "fixed_pre_sizing": {
            "r_t_mm": R_T_FIXED * 1e3, "eps": EPS_FIXED,
            "R_chamber_mm": R_CHAMBER_FIXED * 1e3,
        },
        "operating_point": {
            "p0_Pa": P0_OPER, "T0_K": T0_OPER,
            "W_kg_mol": W_OPER, "p_amb_Pa": P_AMB,
        },
        "design_variables": {
            "bell_fraction": bell_fraction,
            "L_parab_mm": float(result["L [m]"] * 1e3),
            "theta_in_deg": theta_in, "theta_sub_deg": theta_sub,
        },
        "geometry": {
            "r_e_mm": float(result["r_e [m]"] * 1e3),
            "theta_out_deg": float(result["theta_out [°]"]),
            "L_conv_mm": float(result["L_conv [m]"] * 1e3),
        },
        "performance": {
            "Isp_s": float(result["Isp [s]"]), "Isp_ref_s": ISP_REF,
            "fitness": float(fitness), "eta_div": float(result["eta_div"]),
            "eta_visc_sup": float(result["eta_BL_sup"]),
            "eta_BL_sub": float(result["eta_BL_sub"]),
            "friction_force_N": float(result["friction_force [N]"]),
            "eta_total": float(result["eta_total"]),
        },
    }
    path = os.path.join(RESULTS_DIR, f"resultados_isp_{STAMP}.yaml")
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("GA bell nozzle: r_t, eps e R_chamber fixos pelo pré-sizing")
    print(f"r_t={R_T_FIXED*1e3:.3f} mm | eps={EPS_FIXED:.3f} | "
          f"R_chamber={R_CHAMBER_FIXED*1e3:.1f} mm")
    ga = build_ga()
    ga.run()
    solution, fitness, _ = ga.best_solution()
    _, result = evaluate_solution(solution, verbose=True)
    print(f"Solução: bell={solution[0]*100:.2f}% | theta_in={solution[1]:.3f}° | "
          f"theta_sub={solution[2]:.3f}° | theta_out={result['theta_out [°]']:.3f}°")
    print(f"Isp={result['Isp [s]']:.3f} s | fitness={fitness:.6f}")
    print(f"Plot: {save_plots(ga)}")
    print(f"YAML: {save_yaml(solution, result, fitness)}")


if __name__ == "__main__":
    main()
