"""
Fitness Function para Optimizador Genético — Bell Nozzle

PEC Otimização de componente do sistema propulsor, Invictus III / FEUP
Autor: Francisco Ferreira (up202306497)

Integra compute_nozzle_efficiency do módulo apropulsive_performance_model
como função de mérito para o GA (PyGAD).

Variáveis de decisão (7) — vector `solution`:
    [0] r_t           : raio da garganta [m]
    [1] eps           : coeficiente de expansão Ae/At [-]
    [2] theta_in_deg  : ângulo de entrada supersónico [°]
    [3] theta_out_deg : ângulo de saída [°]
    [4] alpha_deg     : ângulo de referência cónico [°]
    [5] theta_sub_deg : ângulo da linha recta convergente [°]
    [6] R_chamber     : raio da câmara de combustão [m]

Função de mérito:
    merito = eta_total - pen_geometria - pen_curvatura - pen_BL_garganta
    (limitado inferiormente a 0)

Penalizações:
    pen_r_e     : raio de saída r_e > 40 mm
    pen_L_div   : comprimento divergente L > 200 mm
    pen_L_conv  : comprimento convergente L_conv > 150 mm  (novo)
    pen_L_total : comprimento total L_div + L_conv > 300 mm  (novo)
    pen_curv    : coeficiente parabólico a < -0.05  (contorno côncavo)
    pen_BL      : desvio de área efectiva de garganta > 2 %
"""

from apropulsive_performance_model import compute_nozzle_efficiency
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# LIMITES DAS PENALIZAÇÕES  
# ═══════════════════════════════════════════════════════════════════════════════

LIM_RE_MAX      = 0.090   # raio de saída máximo [m]  (90 mm)
LIM_L_DIV_MAX   = 0.140   # comprimento divergente máximo [m]  (140 mm)
LIM_L_DIV_MIN   = 0.050   # comprimento divergente mínimo [m]  (50 mm)  
LIM_L_CONV_MAX  = 0.100   # comprimento convergente máximo [m]  (100 mm)
LIM_L_TOT_MAX   = LIM_L_DIV_MAX + LIM_L_CONV_MAX  # comprimento total máximo [m]  

LIM_A_MIN       = -15.0   # limiar de curvatura excessiva da parábola [m⁻¹]
LIM_A_MAX       = -3.0    # limiar de curvatura demasiado linear da parábola [m⁻¹]  
LIM_BL_DESVIO   = 0.02    # desvio máximo tolerado de área efectiva de garganta [-]

K_PEN           = 100.0   # factor de escala das penalizações geométricas
K_PEN_BL        = 20.0    # factor de escala da penalização de camada limite


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def fitness_function(ga_instance, solution, solution_idx):
    """
    Função de mérito para o GA (interface PyGAD).

    Parâmetros
    ----------
    ga_instance  : instância PyGAD (não utilizado internamente)
    solution     : array com os 7 parâmetros de decisão
    solution_idx : índice da solução na população (não utilizado)

    Retorna
    -------
    merito : float ∈ [0, 1]
        0  → solução inválida ou fortemente penalizada
        ~1 → solução próxima do óptimo geométrico
    """
    r_t, eps, theta_in_deg, theta_out_deg, alpha_deg, theta_sub_deg, R_chamber = solution

    try:
        res = compute_nozzle_efficiency(
            r_t           = r_t,
            eps           = eps,
            theta_in_deg  = theta_in_deg,
            theta_out_deg = theta_out_deg,
            alpha_deg     = alpha_deg,
            theta_sub_deg = theta_sub_deg,
            R_chamber     = R_chamber,
            verbose       = False,
        )

        eta_total = res["eta_total"]


        # ── Rejeitar soluções numericamente inválidas ─────────────────────────
        if not np.isfinite(eta_total) or not (0.0 < eta_total <= 1.0):
            return 0.0

        # ── Penalizações ──────────────────────────────────────────────────────
        pen = 0.0

        # 1. Raio de saída (envelope radial do veículo)
        r_e = res["r_e [m]"]
        if r_e > LIM_RE_MAX:
            pen += (r_e - LIM_RE_MAX) * K_PEN

        # 2. Comprimento da secção divergente
        L_div = res["L [m]"]
        if L_div > LIM_L_DIV_MAX:
            pen += (L_div - LIM_L_DIV_MAX) * K_PEN

        if L_div < LIM_L_DIV_MIN:
            pen += (LIM_L_DIV_MIN - L_div) * K_PEN

        # 3. Comprimento da secção convergente  
        L_conv = res["L_conv [m]"]
        if L_conv > LIM_L_CONV_MAX:
            pen += (L_conv - LIM_L_CONV_MAX) * K_PEN

        # 4. Comprimento total do nozzle  
        L_total = L_div + L_conv
        if L_total > LIM_L_TOT_MAX:
            pen += (L_total - LIM_L_TOT_MAX) * K_PEN

        # 5.a. Curvatura da parábola (a < 0 → contorno côncavo — fisicamente indesejável)
        a = res["a"]
        if a < LIM_A_MIN:
            pen += abs(a - LIM_A_MIN) * K_PEN

        # 5.b. Curvatura da parábola (a > 0 → contorno convexo ou linear — perda de eficiência)
        if a > LIM_A_MAX:
            pen += abs(a - LIM_A_MAX) * K_PEN

        # 6. Bloqueio de camada limite na garganta (Mach = 1 na garganta)
        #    eta_BL_sub = (r_t_eff / r_t)²  →  desvio de área = 1 - eta_BL_sub
        #    Equivalente à penalização original sobre At = (r_t / r_t_eff)²
        desvio_BL = 1.0 - res["eta_BL_sub"]
        if desvio_BL > LIM_BL_DESVIO:
            pen += K_PEN_BL * (desvio_BL - LIM_BL_DESVIO)

        # ── Mérito final ──────────────────────────────────────────────────────
        merito = eta_total - pen
        return max(merito, 0.0)

    except Exception:
        return 0.0