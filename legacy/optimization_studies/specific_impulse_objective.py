"""Fitness do GA para otimização do contorno de um bell nozzle.

O pré-sizing fixa garganta, razão de expansão e câmara. O cromossoma contém:
    [0] bell_fraction  L_div / L_cone,15deg [-]
    [1] theta_in_deg  ângulo da parede no início da parábola [deg]
    [2] theta_sub_deg ângulo da reta convergente [deg]
"""

import numpy as np

try:
    from .reduced_order_nozzle_performance import compute_nozzle_efficiency
except ImportError:  # execução direta a partir da pasta genetic_algorithm
    from reduced_order_nozzle_performance import compute_nozzle_efficiency


# Inputs fixos do pré-sizing. Alterar estes valores para cada motor/missão.
R_T_FIXED = 0.01531
EPS_FIXED = 5.60
R_CHAMBER_FIXED = 0.060

# Condições de operação / propriedades CEA.
P0_OPER = 50e5
T0_OPER = 3181.8
W_OPER = 0.024064
P_AMB = 0.0
ISP_REF = 252.85

# Restrições de envelope. Os limites dos genes tratam do intervalo de projeto;
# estas penalizações protegem contra combinações geometricamente inviáveis.
LIM_RE_MAX = 0.090
LIM_L_DIV_MAX = 0.140
LIM_L_CONV_MAX = 0.100
LIM_L_TOT_MAX = LIM_L_DIV_MAX + LIM_L_CONV_MAX
MAX_EXIT_ANGLE_DEG = 15.0
LIM_BL_DESVIO = 0.02

K_PEN_LENGTH = 100.0
K_PEN_ANGLE = 0.10
K_PEN_BL = 20.0


def evaluate_solution(solution, verbose=False):
    """Avalia um cromossoma e devolve ``(fitness, resultados)``."""
    bell_fraction, theta_in_deg, theta_sub_deg = map(float, solution)

    res = compute_nozzle_efficiency(
        r_t=R_T_FIXED,
        eps=EPS_FIXED,
        theta_in_deg=theta_in_deg,
        bell_fraction=bell_fraction,
        theta_sub_deg=theta_sub_deg,
        R_chamber=R_CHAMBER_FIXED,
        p0=P0_OPER,
        T0_prop=T0_OPER,
        W=W_OPER,
        p_amb=P_AMB,
        verbose=verbose,
    )

    isp = res.get("Isp [s]", np.nan)
    if not np.isfinite(isp) or isp <= 0.0:
        return 0.0, res

    penalty = 0.0
    r_e = res["r_e [m]"]
    L_div = res["L [m]"]
    L_conv = res["L_conv [m]"]
    theta_e = res["theta_out [°]"]

    if r_e > LIM_RE_MAX:
        penalty += K_PEN_LENGTH * (r_e - LIM_RE_MAX)
    if L_div > LIM_L_DIV_MAX:
        penalty += K_PEN_LENGTH * (L_div - LIM_L_DIV_MAX)
    if L_conv > LIM_L_CONV_MAX:
        penalty += K_PEN_LENGTH * (L_conv - LIM_L_CONV_MAX)
    if L_div + L_conv > LIM_L_TOT_MAX:
        penalty += K_PEN_LENGTH * (L_div + L_conv - LIM_L_TOT_MAX)
    if theta_e > MAX_EXIT_ANGLE_DEG:
        penalty += K_PEN_ANGLE * (theta_e - MAX_EXIT_ANGLE_DEG)

    blockage = 1.0 - res["eta_BL_sub"]
    if blockage > LIM_BL_DESVIO:
        penalty += K_PEN_BL * (blockage - LIM_BL_DESVIO)

    return max(isp / ISP_REF - penalty, 0.0), res


def fitness_function(ga_instance, solution, solution_idx):
    """Interface exigida pelo PyGAD; soluções inválidas recebem fitness zero."""
    try:
        fitness, _ = evaluate_solution(solution, verbose=False)
        return fitness
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        return 0.0
