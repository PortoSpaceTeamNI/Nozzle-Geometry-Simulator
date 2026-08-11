"""
Fitness Function para Optimizador Genético — Bell Nozzle
Mérito: Impulso Específico (Isp)

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
    merito = (Isp / ISP_REF) - pen_geometria - pen_curvatura - pen_BL_garganta
    (limitado inferiormente a 0)

    ISP_REF é um valor de referência (Isp cónico ideal a 15°) usado para
    normalizar o mérito para a escala [0, ~1], mantendo compatibilidade com
    PyGAD e comparabilidade entre gerações.

Penalizações (idênticas às da fitness_function_final.py):
    pen_r_e     : raio de saída r_e > 60 mm
    pen_L_div   : comprimento divergente L > 150 mm
    pen_L_conv  : comprimento convergente L_conv > 100 mm
    pen_L_total : comprimento total L_div + L_conv > 220 mm
    pen_curv    : coeficiente parabólico a fora de [LIM_A_MIN, LIM_A_MAX]
    pen_BL      : desvio de área efectiva de garganta > 2 %

Inputs termodinâmicos (fixos — condições de operação Invictus III):
    P0_OPER   : pressão de câmara [Pa]
    T0_OPER   : temperatura de câmara [K]
    W_OPER    : massa molar média dos gases [kg/mol]
    P_AMB     : pressão ambiente na altitude de operação [Pa]
"""

from apropulsive_performance_model import compute_nozzle_efficiency
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# INPUTS TERMODINÂMICOS DE OPERAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
# Substituir pelos valores CEA do propulsor real (HTPB/N2O — Invictus III).

P0_OPER  = 50e5      # pressão de estagnação na câmara [Pa]  (25 bar)
T0_OPER  = 3181.8    # temperatura de estagnação na câmara [K]
W_OPER   = 0.024064    # massa molar média dos gases [kg/mol]  (22 g/mol)
P_AMB    = 0.0       # pressão ambiente: vácuo [Pa]  (alterar p/ nível do mar se necessário)

# ═══════════════════════════════════════════════════════════════════════════════
# REFERÊNCIA DE NORMALIZAÇÃO DO ISP
# ═══════════════════════════════════════════════════════════════════════════════
# Isp teórico de referência [s] usado para normalizar o mérito.
# Calculado offline para o nozzle cónico ideal a 15° com os inputs acima.
# Actualizar se P0_OPER / T0_OPER / W_OPER mudarem.

ISP_REF  = 252.85     # [s]  — valor indicativo; actualizar com CEA real


# ═══════════════════════════════════════════════════════════════════════════════
# LIMITES DAS PENALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

LIM_RE_MAX      = 0.090   # raio de saída máximo [m]  (90 mm)
LIM_L_DIV_MAX   = 0.140   # comprimento divergente máximo [m]  (140 mm)
LIM_L_CONV_MAX  = 0.100   # comprimento convergente máximo [m]  (100 mm)
LIM_L_TOT_MAX   = LIM_L_CONV_MAX + LIM_L_DIV_MAX  # comprimento total máximo [m]  

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
    Função de mérito para o GA (interface PyGAD) — maximização do Isp.

    O Isp é normalizado por ISP_REF de modo a que o mérito seja adimensional
    e comparável entre gerações. Um nozzle ideal a 15° (cónico, sem perdas)
    devolveria merito ≈ 1.0; bell nozzles optimizados superam tipicamente
    esse valor em 1–3 %.

    Parâmetros
    ----------
    ga_instance  : instância PyGAD (não utilizado internamente)
    solution     : array com os 7 parâmetros de decisão
    solution_idx : índice da solução na população (não utilizado)

    Retorna
    -------
    merito : float ∈ [0, ~1.05]
        0    → solução inválida ou fortemente penalizada
        ~1   → Isp próximo de ISP_REF (referência cónica ideal)
        >1   → bell nozzle supera a referência cónica (objectivo)
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
            # ── Inputs termodinâmicos ─────────────────────────────────────────
            p0            = P0_OPER,
            T0_prop       = T0_OPER,
            W             = W_OPER,
            p_amb         = P_AMB,
            verbose       = False,
        )

        # ── Verificar se Isp foi calculado ────────────────────────────────────
        if "Isp [s]" not in res:
            return 0.0

        Isp = res["Isp [s]"]

        # ── Rejeitar soluções numericamente inválidas ─────────────────────────
        if not np.isfinite(Isp) or Isp <= 0.0:
            return 0.0

        # ── Normalização ──────────────────────────────────────────────────────
        # Mérito base: Isp relativo ao valor de referência cónico.
        # Mantém a escala próxima de 1.0 e evita que o GA trabalhe com
        # magnitudes absolutas (~200-300 s), o que dificultaria a convergência.
        merito_base = Isp / ISP_REF

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

        # 3. Comprimento da secção convergente
        L_conv = res["L_conv [m]"]
        if L_conv > LIM_L_CONV_MAX:
            pen += (L_conv - LIM_L_CONV_MAX) * K_PEN

        # 4. Comprimento total do nozzle
        L_total = L_div + L_conv
        if L_total > LIM_L_TOT_MAX:
            pen += (L_total - LIM_L_TOT_MAX) * K_PEN

        # 5.a. Curvatura excessiva da parábola (contorno côncavo)
        a = res["a"]
        if a < LIM_A_MIN:
            pen += abs(a - LIM_A_MIN) * K_PEN

        # 5.b. Curvatura insuficiente da parábola (contorno convexo / linear)
        if a > LIM_A_MAX:
            pen += abs(a - LIM_A_MAX) * K_PEN

        # 6. Bloqueio de camada limite na garganta
        #    desvio_BL = 1 - eta_BL_sub  →  fracção de área de garganta perdida
        desvio_BL = 1.0 - res["eta_BL_sub"]
        if desvio_BL > LIM_BL_DESVIO:
            pen += K_PEN_BL * (desvio_BL - LIM_BL_DESVIO)

        # ── Mérito final ──────────────────────────────────────────────────────
        merito = merito_base - pen
        return max(merito, 0.0)

    except Exception:
        return 0.0