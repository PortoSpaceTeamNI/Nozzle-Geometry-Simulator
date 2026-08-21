"""
Optimizador Genético — Bell Nozzle

PEC Otimização de componente do sistema propulsor, Invictus III / FEUP
Autor: Francisco Ferreira (up202306497)

Utiliza PyGAD para maximizar o mérito geométrico do bell nozzle definido
em fitness_function_total_efficiency.py, com base no modelo propulsivo de
apropulsive_performance_model.py.

Pós-processamento incluído:
    1. Evolução do melhor indivíduo por geração
    2. Mérito médio, máximo e desvio padrão por geração
    3. Top-N melhores geometrias finais (contornos sobrepostos)
    4. Exportação YAML de resultados
"""

import pygad
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
try:
    from .total_efficiency_objective import fitness_function
    from .reduced_order_nozzle_performance import (
        compute_nozzle_efficiency,
        compute_parabola_coefficients,
        compute_convergent_geometry,
    )
except ImportError:
    from total_efficiency_objective import fitness_function
    from reduced_order_nozzle_performance import (
        compute_nozzle_efficiency,
        compute_parabola_coefficients,
        compute_convergent_geometry,
    )
import yaml
from datetime import datetime
import os

# ── Pasta de resultados (relativa ao script) ──────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_DIR, "Resultados")
os.makedirs(_RESULTS_DIR, exist_ok=True)

_STAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
caminho = os.path.join(_RESULTS_DIR, f"resultados_nozzle_{_STAMP}.yaml")

# Número de melhores soluções a plotar no painel de geometrias
TOP_N = 5


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUTURAS DE REGISTO POR GERAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

_historico = {
    "best_fitness"  : [],   # melhor mérito de cada geração
    "mean_fitness"  : [],   # mérito médio da população
    "std_fitness"   : [],   # desvio padrão do mérito
    "best_solution" : [],   # melhor solução (7 genes) de cada geração
}


def on_generation(ga_inst):
    """Callback chamado no fim de cada geração — regista estatísticas."""
    fits = np.array(ga_inst.last_generation_fitness)
    _historico["best_fitness"].append(float(np.max(fits)))
    _historico["mean_fitness"].append(float(np.mean(fits)))
    _historico["std_fitness"].append(float(np.std(fits)))

    best_idx = int(np.argmax(fits))
    _historico["best_solution"].append(
        ga_inst.population[best_idx].copy()
    )

    gen = ga_inst.generations_completed
    if gen % 25 == 0 or gen == 1:
        print(f"  Gen {gen:>4} | best={_historico['best_fitness'][-1]:.5f}"
              f" | mean={_historico['mean_fitness'][-1]:.5f}"
              f" | std={_historico['std_fitness'][-1]:.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAR O ALGORITMO GENÉTICO
# ═══════════════════════════════════════════════════════════════════════════════

ga_instance = pygad.GA(
    num_generations        = 300,
    num_parents_mating     = 20,
    fitness_func           = fitness_function,
    sol_per_pop            = 100,
    gene_space = [
        {"low": 0.005, "high": 0.020},   # [0] r_t           [m]    5 mm – 20 mm
        {"low": 3.0,   "high": 7.0},     # [1] eps            [-]    3 – 7
        {"low": 20.0,  "high": 35.0},    # [2] theta_in_deg  [°]    20° – 35°
        {"low": 0.5,   "high": 10.0},    # [3] theta_out_deg [°]     0.5° – 10.0°
        {"low": 10.0,  "high": 18.0},    # [4] alpha_deg     [°]    10° – 18°
        {"low": 40.0,  "high": 65.0},    # [5] theta_sub_deg [°]    40° – 65°
        {"low": 0.040, "high": 0.060},   # [6] R_chamber     [m]    40 mm – 60 mm
    ],
    gene_type              = float,
    num_genes              = 7,
    mutation_percent_genes = [20, 5],
    parent_selection_type  = "tournament",
    crossover_type         = "uniform",
    crossover_probability  = 0.85,
    mutation_type          = "adaptive",
    keep_elitism           = 3,
    stop_criteria          = "saturate_40",
    save_solutions         = True,
    on_generation          = on_generation,   # ← callback de registo
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PÓS-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def _nozzle_contour(sol, n=400):
    """
    Devolve (x_mm, r_mm) do contorno completo: convergente + supersónico.
    Retorna None se a geometria for inválida.
    """
    r_t, eps, theta_in, theta_out, alpha, theta_sub, R_c = sol
    try:
        # Secção convergente
        x_conv, r_conv, _ = compute_convergent_geometry(r_t, theta_sub, R_c)

        # Arco supersónico + parábola divergente
        a, b, c, xP, L = compute_parabola_coefficients(r_t, eps, theta_in, alpha)
        x_div = np.linspace(xP, L, n)
        r_div = a * x_div**2 + b * x_div + c

        # Juntar (garganta em x=0 é ponto de ligação)
        x_full = np.concatenate([x_conv, x_div]) * 1e3   # → mm
        r_full = np.concatenate([r_conv, r_div]) * 1e3
        return x_full, r_full
    except Exception:
        return None


def plot_postprocessing(ga_inst, top_n=TOP_N):
    """
    Generates and saves the post-processing panel:
        B  Population standard deviation over generations
        C  Normalised evolution of best individual parameters
        D  Top-N best final nozzle contours (full geometry)
    """
    n_gen  = len(_historico["best_fitness"])
    gens   = np.arange(1, n_gen + 1)
    best   = np.array(_historico["best_fitness"])
    mean   = np.array(_historico["mean_fitness"])
    std    = np.array(_historico["std_fitness"])

    # ── Top-N final solutions ─────────────────────────────────────────────────
    fits_final = np.array(ga_inst.last_generation_fitness)
    top_idx    = np.argsort(fits_final)[::-1][:top_n]
    top_sols   = [ga_inst.population[i] for i in top_idx]
    top_fits   = [fits_final[i] for i in top_idx]

    # ── Parameter names ───────────────────────────────────────────────────────
    param_names  = [r"$r_t$ [mm]", r"$\varepsilon$",
                    r"$\theta_{in}$ [°]", r"$\theta_{out}$ [°]",
                    r"$\alpha$ [°]", r"$\theta_{sub}$ [°]",
                    r"$R_c$ [mm]"]
    param_scales = [1e3, 1, 1, 1, 1, 1, 1e3]

    best_sols_arr = np.array(_historico["best_solution"])   # (n_gen, 7)

    # ── Layout: narrow left column (2 plots), wide right column (nozzle) ─────
    fig = plt.figure(figsize=(20, 12), facecolor="white")
    gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.92, bottom=0.12,
                           width_ratios=[1, 2.2])

    CMAP   = plt.cm.plasma
    colors = [CMAP(i / max(top_n - 1, 1)) for i in range(top_n)]

    FS_TITLE  = 13   # title fontsize
    FS_LABEL  = 12   # axis label fontsize
    FS_TICK   = 11   # tick label fontsize
    FS_LEGEND = 10   # legend fontsize

    # ── B: Population standard deviation over generations ────────────────────
    ax_b = fig.add_subplot(gs[0, 0])
    ax_b.plot(gens, std, color="#d62728", lw=2.0, label="Standard deviation")
    ax_b.fill_between(gens, std, alpha=0.15, color="#d62728")
    ax_b.set_xlabel("Generation", fontsize=FS_LABEL)
    ax_b.set_ylabel("Fitness standard deviation", fontsize=FS_LABEL)
    ax_b.set_title("Population Dispersion over Generations",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_b.tick_params(labelsize=FS_TICK)
    ax_b.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax_b.legend(fontsize=FS_LEGEND)

    # ── C: Normalised evolution of best individual parameters ─────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for j, (name, scale) in enumerate(zip(param_names, param_scales)):
        vals = best_sols_arr[:, j] * scale
        lo, hi = vals.min(), vals.max()
        vals_norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        ax_c.plot(gens, vals_norm, lw=1.6, label=name)

    ax_c.set_xlabel("Generation", fontsize=FS_LABEL)
    ax_c.set_ylabel("Normalised value [0–1]", fontsize=FS_LABEL)
    ax_c.set_title("Best Individual Parameter Evolution",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_c.tick_params(labelsize=FS_TICK)
    ax_c.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax_c.legend(fontsize=FS_LEGEND, ncol=4, loc="upper center",
                bbox_to_anchor=(0.5, -0.20), frameon=True)

    # ── D: Top-N contours — full height right column ──────────────────────────
    ax_d = fig.add_subplot(gs[:, 1])
    for rank, (sol, fit, col) in enumerate(zip(top_sols, top_fits, colors)):
        contour = _nozzle_contour(sol)
        if contour is None:
            continue
        x_mm, r_mm = contour
        lw    = 2.5 if rank == 0 else 1.4
        ls    = "-"  if rank == 0 else "--"
        label = f"#{rank+1}  fit={fit:.4f}"
        ax_d.plot(x_mm, r_mm, color=col, lw=lw, ls=ls, label=label)

    ax_d.set_ylim(bottom=0)
    ax_d.set_xlabel("Axial position [mm]", fontsize=FS_LABEL)
    ax_d.set_ylabel("Radius [mm]", fontsize=FS_LABEL)
    ax_d.set_title(f"Top-{top_n} Best Final Geometries",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_d.tick_params(labelsize=FS_TICK)
    ax_d.set_aspect("equal")
    ax_d.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax_d.legend(fontsize=FS_LEGEND, loc="upper center",
                bbox_to_anchor=(0.5, -0.06), frameon=True, ncol=1)

    fig.suptitle("Post-processing — GA Bell Nozzle Optimisation (Total Efficiency)",
                 fontsize=15, fontweight="bold", y=0.97)

    path_fig = os.path.join(_RESULTS_DIR, f"postprocessing_{_STAMP}.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Post-processing plot saved: {path_fig}")
    return path_fig

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GUARDAR RESULTADOS YAML
# ═══════════════════════════════════════════════════════════════════════════════

def guardar_resultados_yaml(solution, resultado, fitness):
    r_t, eps, theta_in_deg, theta_out_deg, alpha_deg, theta_sub_deg, R_chamber = solution

    # Estatísticas finais da convergência
    best_arr = np.array(_historico["best_fitness"])
    mean_arr = np.array(_historico["mean_fitness"])
    std_arr  = np.array(_historico["std_fitness"])

    dados = {
        "metadata": {
            "data"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "projeto": "Invictus III / FEUP",
            "metodo" : "Bell Nozzle — Algoritmo Genético (PyGAD)",
        },
        "variaveis_decisao": {
            "r_t_mm"        : float(r_t * 1e3),
            "eps"           : float(eps),
            "theta_in_deg"  : float(theta_in_deg),
            "theta_out_deg" : float(theta_out_deg),
            "alpha_deg"     : float(alpha_deg),
            "theta_sub_deg" : float(theta_sub_deg),
            "R_chamber_mm"  : float(R_chamber * 1e3),
        },
        "geometria": {
            "r_e_mm"                : float(resultado["r_e [m]"] * 1e3),
            "L_div_mm"              : float(resultado["L [m]"] * 1e3),
            "L_conv_mm"             : float(resultado["L_conv [m]"] * 1e3),
            "L_total_mm"            : float((resultado["L [m]"] + resultado["L_conv [m]"]) * 1e3),
            "theta_out_deg"         : float(resultado["theta_out [°]"]),
            "theta_out_parabola_deg": float(resultado["theta_out_parabola [°]"]),
            "r_t_eff_mm"            : float(resultado["r_t_eff [m]"] * 1e3),
            "delta_star_throat_um"  : float(resultado["delta_star_throat [m]"] * 1e6),
            "a"                     : float(resultado["a"]),
            "b"                     : float(resultado["b"]),
            "c"                     : float(resultado["c"]),
        },
        "rendimentos": {
            "eta_div"               : float(resultado["eta_div"]),
            "eta_BL_sup"            : float(resultado["eta_BL_sup"]),
            "eta_BL_sub"            : float(resultado["eta_BL_sub"]),
            "eta_turn"              : float(resultado["eta_turn"]),
            "eta_total"             : float(resultado["eta_total"]),
            "eta_total_percentagem" : float(resultado["eta_total"] * 100),
        },
        "convergencia": {
            "geracoes_corridas"     : len(best_arr),
            "fitness_final"         : float(fitness),
            "fitness_maximo"        : float(best_arr.max()),
            "fitness_medio_final"   : float(mean_arr[-1]),
            "std_final"             : float(std_arr[-1]),
            "geracao_convergencia"  : int(np.argmax(best_arr) + 1),
        },
        "ga_config": {
            "num_generations"       : ga_instance.num_generations,
            "sol_per_pop"           : ga_instance.sol_per_pop,
            "num_parents_mating"    : ga_instance.num_parents_mating,
            "mutation_percent_genes": ga_instance.mutation_percent_genes,
            "keep_elitism"          : ga_instance.keep_elitism,
            "fitness_final"         : float(fitness),
        },
    }

    print(dados)
    with open(caminho, "w", encoding="utf-8") as f:
        yaml.dump(dados, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\nResultados YAML guardados em: {caminho}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXECUÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("A iniciar otimização do bell nozzle...")
    ga_instance.run()

    solution, fitness, _ = ga_instance.best_solution()
    r_t, eps, theta_in_deg, theta_out_deg, alpha_deg, theta_sub_deg, R_chamber = solution

    print("\n>>> SOLUÇÃO ÓTIMA ENCONTRADA:")
    resultado = compute_nozzle_efficiency(
        r_t           = r_t,
        eps           = eps,
        theta_in_deg  = theta_in_deg,
        theta_out_deg = theta_out_deg,
        alpha_deg     = alpha_deg,
        theta_sub_deg = theta_sub_deg,
        R_chamber     = R_chamber,
        verbose       = True,
    )

    # ── Pós-processamento ─────────────────────────────────────────────────────
    # Plot nativo do PyGAD (evolução do mérito) guardado em ficheiro
    ga_instance.plot_fitness(
        title    = "Evolução do mérito",
        ylabel   = "fitness (mérito)",
        save_dir = os.path.join(_RESULTS_DIR, f"fitness_evolution_{_STAMP}.png"),
    )
    plt.close("all")

    plot_postprocessing(ga_instance, top_n=TOP_N)
    guardar_resultados_yaml(solution, resultado, fitness)
