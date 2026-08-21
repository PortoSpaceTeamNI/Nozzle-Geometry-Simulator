"""
Propulsive Performance Model para Bell Nozzle

PEC Otimização de componente do sistema propulsor, Invictus III / FEUP
Autor: Francisco Ferreira (up202306497)

Calcula o rendimento total de um bell nozzle a partir de:
    η_total = η_div · η_visc_sup · η_BL_sub

e, opcionalmente, as métricas propulsivas completas (Sec. 3 do doc. de referência):
    F   : thrust [N]          
    Isp : impulso específico [s] 
    J   : impulso total [N·s] 

Variáveis geométricas:
    r_t       : raio da garganta [m]
    eps       : coeficiente de expansão (ε = Ae/At) [-]
    theta_in  : ângulo de entrada supersónico [°]
    bell_fraction : L_div/L_cone,15deg [-]
    theta_sub : ângulo da linha recta convergente [°]
    R_chamber  : raio da câmara de combustão [m]

Inputs termodinâmicos adicionais (para thrust/Isp):
    p0        : pressão de estagnação na câmara [Pa]
    W         : massa molar média dos gases [kg/mol]
    p_amb     : pressão ambiente na altitude de operação [Pa]
    t_burn    : duração de queima [s]  (para impulso total J)

Dependências: numpy, scipy
"""

import numpy as np
import sys
from scipy.optimize import brentq


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARÂMETROS DO GÁS
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA   = 1.1375     # rácio de calores específicos
R_GAS   = 345.52     # constante do gás [J/(kg·K)]
T0      = 3191.8     # temperatura de estagnação [K]
MU      = 8.643e-5   # viscosidade dinâmica [Pa·s]


G0      = 9.80665    # aceleração gravítica padrão [m/s²]  (ISO 80000-3)
R_UNIV  = 8.31446    # constante universal dos gases [J/(mol·K)]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GEOMETRIA DO NOZZLE
# ═══════════════════════════════════════════════════════════════════════════════

# 2.1 GEOMETRIA DIVERGENTE

def compute_exit_radius(r_t: float, eps: float) -> float:
    """Raio de saída (eq. 1)."""
    return np.sqrt(eps) * r_t


def compute_nozzle_length(r_t: float, eps: float, bell_fraction: float,
                          reference_angle_deg: float = 15.0) -> float:
    """Comprimento divergente como fração do cone equivalente de 15 graus."""
    r_e    = compute_exit_radius(r_t, eps)
    L_cone = (r_e - r_t) / np.tan(np.radians(reference_angle_deg))
    return bell_fraction * L_cone


def compute_supersonic_arc_point(r_t: float):
    """
    Ponto de início da parábola supersónica (xP, yP).
    Arco supersónico: centro (0, 1.4·r_t), raio 0.4·r_t, θ_i = 30°.
    """
    xP = 0.4 * r_t * np.sin(np.radians(30))
    yP = 1.4 * r_t - np.sqrt((0.4 * r_t)**2 - xP**2)
    return xP, yP


def compute_parabola_coefficients(r_t, eps, theta_in, bell_fraction):
    """Coeficientes (a, b, c) da parábola supersónica (eqs. 3-6)."""
    xP, yP = compute_supersonic_arc_point(r_t)
    r_e    = compute_exit_radius(r_t, eps)
    L      = compute_nozzle_length(r_t, eps, bell_fraction)

    A_mat = np.array([
        [xP**2, xP, 1],
        [2*xP,   1, 0],
        [L**2,   L, 1],
    ])
    rhs = np.array([yP, np.tan(np.radians(theta_in)), r_e])
    a, b, c = np.linalg.solve(A_mat, rhs)
    return a, b, c, xP, L


# 2.2 GEOMETRIA CONVERGENTE

def compute_convergent_geometry(r_t: float,
                                 theta_sub_deg: float,
                                 R_chamber: float):

    theta_sub = np.radians(theta_sub_deg)

    # Ponto Q: interseção linha recta / arco subsónico
    xr = -1.5 * r_t * np.sin(theta_sub)
    yr =  2.5 * r_t - 1.5 * r_t * np.cos(theta_sub)

    # Linha recta: y(xf) = R_chamber  →  xf
    m  = -np.tan(theta_sub)
    b  =  yr - m * xr          # y = m·x + b
    xf = (R_chamber - b) / m   # xf < xr < 0

    n_pts = 300

    # Segmento 1: linha recta  [xf, xr]
    x0 = np.linspace(xf, xr, n_pts // 2)
    r0 = m * x0 + b

    # Segmento 2: arco subsónico  [xr, 0]
    x1 = np.linspace(xr, 0.0, n_pts // 2)
    r1 = 2.5 * r_t - np.sqrt(np.maximum((1.5 * r_t)**2 - x1**2, 0.0))

    x_arr = np.concatenate([x0, x1[1:]])   # evitar duplicar xr
    r_arr = np.concatenate([r0, r1[1:]])

    # Comprimento axial total da secção convergente: |xf| (xf < 0, garganta em x=0)
    L_conv = abs(xf)

    return x_arr, r_arr, L_conv


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RELAÇÕES ISENTRÓPICAS E NÚMERO DE MACH
# ═══════════════════════════════════════════════════════════════════════════════

def area_mach_residual(M, area_ratio, gamma=GAMMA):
    """Resíduo F(M; α) = A/A* – α  (eq. 12)."""
    term = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2)
    return (1 / M) * term**((gamma + 1) / (2 * (gamma - 1))) - area_ratio


def mach_from_area_ratio(area_ratio, supersonic=True, gamma=GAMMA):
    """
    Resolve a relação área-Mach (eq. 11).
    supersonic=True  → ramo supersónico (M > 1)
    supersonic=False → ramo subsónico   (M < 1)
    """
    if area_ratio <= 1.0:
        return 1.0
    if supersonic:
        M_lo, M_hi = 1.0 + 1e-9, 50.0
    else:
        M_lo, M_hi = 1e-9, 1.0 - 1e-9
    try:
        M = brentq(area_mach_residual, M_lo, M_hi,
                   args=(area_ratio,), xtol=1e-10, rtol=1e-10)
    except ValueError:
        M = np.nan
    return M


def exit_velocity(Me, T0=T0, gamma=GAMMA, R=R_GAS):
    """Velocidade de saída (eq. 23)."""
    Te = T0 / (1 + (gamma - 1) / 2 * Me**2)
    return Me * np.sqrt(gamma * R * Te)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RENDIMENTOS INDIVIDUAIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4.1  η_div ─────────────────────────────────────────────────────────────────

def eta_divergence(theta_out):
    """Correção axisimétrica de divergência com o ângulo real de saída."""
    return 0.5 * (1.0 + np.cos(np.radians(theta_out)))


# ── 4.2  η_BL_sup : Camada limite na secção SUPERSÓNICA (eq. 22) ──────────────

def dynamic_viscosity(T, T_ref=T0, mu_ref=MU):
    """Viscosidade dos gases quentes por uma lei de potência simples."""
    return mu_ref * np.asarray(T / T_ref) ** 0.7


def compute_BL_supersonic(r_t, eps, theta_in, bell_fraction, p0,
                           T0=T0, gamma=GAMMA, R=R_GAS, mu_ref=MU,
                           n_points=300):
    """
    Rendimento de camada limite na secção supersónica.
        η_BL_sup = V_e,eff / V_e,ist

    A tensão de corte é integrada na parede, incluindo a área molhada.
    """
    a, b, c, xP, L = compute_parabola_coefficients(
        r_t, eps, theta_in, bell_fraction)
    A_t = np.pi * r_t**2

    x_arr  = np.linspace(xP, L, n_points)
    r_arr  = a * x_arr**2 + b * x_arr + c
    A_arr  = np.pi * r_arr**2
    AR_arr = A_arr / A_t

    M_arr  = np.array([mach_from_area_ratio(ar, supersonic=True) for ar in AR_arr])
    T_arr  = T0 / (1 + (gamma - 1) / 2 * M_arr**2)
    a_arr  = np.sqrt(gamma * R * T_arr)
    u_arr  = M_arr * a_arr
    p_arr = p0 * (1 + (gamma - 1) / 2 * M_arr**2)**(-gamma / (gamma - 1))
    rho_arr = p_arr / (R * T_arr)
    mu_arr = dynamic_viscosity(T_arr, T_ref=T0, mu_ref=mu_ref)

    slope = 2.0 * a * x_arr + b
    ds_dx = np.sqrt(1.0 + slope**2)
    s_arr = np.zeros_like(x_arr)
    s_arr[1:] = np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1]) * np.diff(x_arr))
    s_safe = np.maximum(s_arr + 0.4 * r_t, 1e-8)
    Re_s = np.maximum(rho_arr * u_arr * s_safe / mu_arr, 1.0)
    cf = np.where(Re_s < 5.0e5, 0.664 / np.sqrt(Re_s), 0.0592 / Re_s**0.2)
    tau_w = 0.5 * cf * rho_arr * u_arr**2

    # Para a projeção axial, dS*cos(theta) = 2*pi*r*dx.
    friction_force = np.trapezoid(tau_w * 2.0 * np.pi * r_arr, x_arr)
    mdot_ideal = p0 * A_t * vandenkerckhove(gamma) / np.sqrt(R * T0)
    Me_ideal = mach_from_area_ratio(eps, supersonic=True, gamma=gamma)
    Ve_ideal = exit_velocity(Me_ideal, T0, gamma, R)
    loss_fraction = friction_force / max(mdot_ideal * Ve_ideal, 1e-12)
    eta = float(np.clip(1.0 - loss_fraction, 0.0, 1.0))
    return eta, float(friction_force), float(loss_fraction)


# ── 4.3  η_BL_sub : Camada limite na secção CONVERGENTE (novo) ────────────────

def compute_BL_convergent(r_t, eps, theta_sub_deg, R_chamber, p0,
                           T0=T0, gamma=GAMMA, R=R_GAS, mu=MU):
    """
    Rendimento de camada limite na secção CONVERGENTE.

    Motivação física
    ----------------
    Na secção convergente o escoamento é subsónico e acelera em direção à
    garganta. A camada limite cresce ao longo da parede, reduzindo a área de
    escoamento efectiva. Na garganta, a redução de raio:

        r_t,eff = r_t – δ*(x=0⁻)

    implica uma área de garganta efectiva menor, o que reduz o débito e,
    consequentemente, a velocidade de saída.

    Modelação
    ---------
    • Perfil geométrico subsónico r(x): linha recta + arco subsónico
      (Sec. 2.3 do simulador, convenção x ∈ [xf, 0]).
    • M(x) obtido da relação área-Mach, ramo subsónico, usando A_t como A*.
    • Espessura de deslocamento turbulenta (eq. 18-19 do GA doc):
          δ*(x) = 0.046 |x| / Re_{|x|}^(1/5)
      onde x é medido a partir da garganta (x=0) em direção à câmara,
      portanto a distância de corrida é |x|.
    • Raio efectivo na garganta:
          r_t,eff = r_t – δ*(0⁻)   [limite ao aproximar a garganta]
    • Rendimento:
          η_BL_sub = V_e,eff_sub / V_e,ist
      onde V_e,eff_sub é calculado com a área efectiva de garganta
      (expansão dada pelo mesmo ε mas referida a A_t,eff).

    Nota sobre o sentido de corrente
    ---------------------------------
    A distância de corrente para o crescimento de BL é medida desde onde o
    escoamento "entra" na conduta convergente (x = xf). Usamos portanto:
        s(x) = x – xf   (distância desde a entrada da secção convergente)
    Para que δ* seja calculado com a distância certa a partir do ponto de
    início da camada limite (início da linha recta, onde θ_sub começa).

    Parâmetros
    ----------
    r_t          : raio da garganta [m]
    eps          : coeficiente de expansão [-]
    theta_sub_deg: ângulo da linha recta convergente [°]
    R_chamber    : raio da câmara de combustão [m]

    Retorna
    -------
    eta_BL_sub (float) : rendimento de camada limite convergente ∈ (0, 1]
    delta_star_throat  : espessura de deslocamento estimada na garganta [m]
    """
    x_arr, r_arr, L_conv = compute_convergent_geometry(r_t, theta_sub_deg, R_chamber)

    # Comprimento de corrente desde a entrada (xf) — sempre positivo
    xf  = x_arr[0]
    s   = x_arr - xf          # s ≥ 0; s=0 no início, s_max na garganta

    A_t    = np.pi * r_t**2
    A_arr  = np.pi * r_arr**2
    AR_arr = A_arr / A_t      # A/A* — ramo subsónico (A ≥ A_t perto da garganta,
                               # mas A_arr[0] >> A_t junto à câmara)

    # Número de Mach subsónico em cada ponto
    M_arr = np.array([
        mach_from_area_ratio(ar, supersonic=False) if ar > 1.0 else 1.0
        for ar in AR_arr
    ])

    # Propriedades locais isentrópicas
    T_arr   = T0 / (1 + (gamma - 1) / 2 * M_arr**2)
    a_arr   = np.sqrt(gamma * R * T_arr)
    u_arr   = M_arr * a_arr
    p_arr = p0 * (1 + (gamma - 1) / 2 * M_arr**2)**(-gamma / (gamma - 1))
    rho_arr = p_arr / (R * T_arr)

    # Espessura de deslocamento turbulenta ao longo da secção convergente
    # δ*(s) = 0.046 s / Re_s^(1/5),   Re_s = ρ u s / μ
    s_safe     = np.maximum(s, 1e-9)              # evitar s=0 no ponto inicial
    Re_s       = rho_arr * u_arr * s_safe / mu
    delta_star = 0.046 * s_safe / Re_s**(1/5)

    # Valor de δ* na garganta (último ponto, x → 0)
    delta_star_throat = delta_star[-1]

    # Raio efectivo na garganta
    r_t_eff = max(r_t - delta_star_throat, r_t * 0.01)

    # ── Modelo de perda ──────────────────────────────────────────────────────
    #
    # A camada limite convergente bloqueia uma fracção da área de garganta,
    # reduzindo o débito mássico choked:
    #
    #   ṁ_eff = ṁ_ist · (A_t,eff / A_t) = ṁ_ist · (r_t,eff / r_t)²
    #
    # Para condições de estagnação inalteradas (T0, p0), o débito mássico é
    # proporcional a A_t (relação de débito isentrópico). Portanto:
    #
    #   ṁ_eff / ṁ_ist = (r_t,eff / r_t)²  ≡  f_A
    #
    # O empuxo (e a velocidade efectiva de saída) é proporcional a ṁ · Ve.
    # Como Ve é determinado por ε e pelas condições de estagnação (não muda),
    # a perda surge apenas na redução de débito, que se manifesta como:
    #
    #   η_BL_sub = f_A = (r_t,eff / r_t)²
    #
    # Esta definição é análoga à usada na literatura para perdas de bloqueio
    # (e.g. Nickerson et al., "JANNAF BL Losses"):
    #
    #   η_BL = 1 – 2·δ*/r_t  (linearizado para δ* << r_t)
    #
    # A formulação quadrática exacta é mais consistente com as eqs. 20-22
    # do documento de referência.
    # ─────────────────────────────────────────────────────────────────────────

    f_A = (r_t_eff / r_t)**2   # fracção de área efectiva de garganta

    # Alternativa linearizada (comentada — útil para verificação):
    # f_A_lin = 1.0 - 2.0 * delta_star_throat / r_t

    eta = min(f_A, 1.0)        # garantia física (nunca > 1)
    return eta, delta_star_throat, L_conv


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THRUST E IMPULSO ESPECÍFICO  
# ═══════════════════════════════════════════════════════════════════════════════

def vandenkerckhove(gamma=GAMMA):
    """
    Função de Vandenkerckhove Γ  (eq. 9).

    Γ = √γ · [2/(γ+1)]^((γ+1)/(2(γ−1)))

    Parâmetros
    ----------
    gamma : rácio de calores específicos [-]

    Retorna
    -------
    Gamma : float
    """
    return np.sqrt(gamma) * (2 / (gamma + 1))**((gamma + 1) / (2 * (gamma - 1)))


def mass_flow_rate(r_t, p0, T0_val, W, gamma=GAMMA):
    """
    Débito mássico em condições de garganta bloqueada  (eq. 10).

        ṁ = p₀ · At · Γ / √(R_esp · T₀)

    onde R_esp = R_UNIV / W  é a constante específica do gás.

    Parâmetros
    ----------
    r_t   : raio da garganta [m]
    p0    : pressão de estagnação na câmara [Pa]
    T0_val: temperatura de estagnação na câmara [K]
    W     : massa molar média dos gases de combustão [kg/mol]

    Retorna
    -------
    mdot : débito mássico [kg/s]
    """
    At    = np.pi * r_t**2
    R_esp = R_UNIV / W
    Gamma = vandenkerckhove(gamma)
    return p0 * At * Gamma / np.sqrt(R_esp * T0_val)


def exit_pressure(p0, Me, gamma=GAMMA):
    """
    Pressão estática de saída pela relação isentrópica  (eq. 6).

        pe = p₀ · (1 + (γ−1)/2 · Me²)^(−γ/(γ−1))
    """
    return p0 * (1 + (gamma - 1) / 2 * Me**2)**(- gamma / (gamma - 1))


def compute_thrust_isp(r_t, eps, eta_total, Ve_ist,
                       p0, T0_val, W,
                       p_amb=0.0,
                       t_burn=None,
                       gamma=GAMMA):
    """
    Calcula thrust F, impulso específico Isp e impulso total J a partir
    dos outputs de rendimento já computados  (eqs. 20-23).

        Ve       = η_total · Ve,ist                          [eq. 19]
        ṁ        = p₀ · At · Γ / √(R_esp · T₀)             [eq. 10]
        Me       = f(ε)   (ramo supersónico)
        pe       = p₀ · (isentrópico)                       [eq. 6]
        Ae       = ε · At                                    [eq. 3]
        F        = ṁ · Ve + (pe − pamb) · Ae               [eq. 20]
        Isp      = F / (ṁ · g₀)                             [eq. 21]
        J        = F · t_burn                               [eq. 23]

    Parâmetros
    ----------
    r_t      : raio da garganta [m]
    eps      : coeficiente de expansão Ae/At [-]
    eta_total: rendimento total η_total [-]
    Ve_ist   : velocidade de saída isentrópica ideal [m/s]
    p0       : pressão de estagnação na câmara [Pa]
    T0_val   : temperatura de estagnação na câmara [K]
    W        : massa molar média [kg/mol]
    p_amb    : pressão ambiente na altitude de operação [Pa]  (default 0 → vácuo)
    t_burn   : duração de queima [s]  (None → J não calculado)
    gamma    : rácio de calores específicos [-]

    Retorna
    -------
    dict com F [N], Isp [s], mdot [kg/s], Ve [m/s], pe [Pa], J [N·s] (se t_burn)
    """
    At   = np.pi * r_t**2
    Ae   = eps * At

    # Débito mássico choked  (eq. 10)
    mdot = mass_flow_rate(r_t, p0, T0_val, W, gamma)

    # Velocidade de saída corrigida  (eq. 19)
    Ve   = eta_total * Ve_ist

    # Número de Mach e pressão de saída  (eqs. 4, 6)
    Me   = mach_from_area_ratio(eps, supersonic=True, gamma=gamma)
    pe   = exit_pressure(p0, Me, gamma)

    # Thrust  (eq. 20)
    F_momentum = mdot * Ve
    F_pressure = (pe - p_amb) * Ae
    F          = F_momentum + F_pressure

    # Isp  (eq. 21)
    Isp = F / (mdot * G0)

    result = {
        "mdot [kg/s]"       : mdot,
        "Ve [m/s]"          : Ve,
        "Ve_ist [m/s]"      : Ve_ist,
        "Me [-]"            : Me,
        "pe [Pa]"           : pe,
        "p_amb [Pa]"        : p_amb,
        "F_momentum [N]"    : F_momentum,
        "F_pressure [N]"    : F_pressure,
        "F [N]"             : F,
        "Isp [s]"           : Isp,
    }

    if t_burn is not None:
        mp = mdot * t_burn           # massa de propulsor consumida  (eq. 23)
        J  = F * t_burn              # impulso total
        result["t_burn [s]"]  = t_burn
        result["mp [kg]"]     = mp
        result["J [N·s]"]     = J

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RENDIMENTO TOTAL (ESTENDIDO)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_nozzle_efficiency(r_t,
                               eps,
                               theta_in_deg,
                               bell_fraction,
                               theta_sub_deg=45.0,
                               R_chamber=None,
                               # ── Inputs termodinâmicos para thrust/Isp ──
                               p0=None,
                               T0_prop=None,
                               W=None,
                               p_amb=0.0,
                               t_burn=None,
                               verbose=True):
    """
    Calcula o rendimento total do bell nozzle incluindo perdas na secção
    convergente, e opcionalmente thrust F, Isp e impulso total J.

        η_total = η_div · η_visc,sup · η_BL,sub

    Para calcular F e Isp é necessário fornecer p0, T0_prop e W.
    Se omitidos, apenas os rendimentos são calculados.

    Parâmetros
    ----------
    r_t           : raio da garganta [m]
    eps           : coeficiente de expansão Ae/At [-]
    theta_in_deg  : ângulo de entrada supersónico [°]
    bell_fraction : comprimento divergente / cone equivalente de 15° [-]
    theta_sub_deg : ângulo da linha recta convergente [°]  (default 45°)
    R_chamber     : raio da câmara [m]  (default 3·r_t se None)
    p0            : pressão de estagnação na câmara [Pa]  (necessário para F/Isp)
    T0_prop       : temperatura de estagnação [K]         (necessário para F/Isp;
                    se None usa o global T0)
    W             : massa molar média dos gases [kg/mol]  (necessário para F/Isp)
    p_amb         : pressão ambiente na altitude [Pa]  (default 0 → vácuo)
    t_burn        : duração de queima [s]  (None → J não calculado)
    verbose       : imprimir relatório detalhado

    Retorna
    -------
    dict com todos os rendimentos, parâmetros geométricos, e (se p0/W fornecidos)
    F [N], Isp [s], mdot [kg/s], J [N·s].
    """
    if R_chamber is None:
        R_chamber = 3.0 * r_t

    # Temperatura de estagnação efectiva (global T0 como fallback)
    T0_val = T0_prop if T0_prop is not None else T0

    # ── Geometria divergente ──────────────────────────────────────────────────
    r_e  = compute_exit_radius(r_t, eps)
    if p0 is None:
        raise ValueError("p0 é necessário para calcular a perda viscosa dimensional")
    L    = compute_nozzle_length(r_t, eps, bell_fraction)
    a, b, c, xP, _ = compute_parabola_coefficients(
        r_t, eps, theta_in_deg, bell_fraction)

    theta_out_calc_deg = np.degrees(np.arctan(2 * a * L + b))

    # ── Rendimentos ───────────────────────────────────────────────────────────
    if theta_out_calc_deg < 0.0 or theta_out_calc_deg >= theta_in_deg:
        raise ValueError("contorno inválido: ângulo de saída fora de [0, theta_in)")

    n_div = eta_divergence(theta_out_calc_deg)
    n_BL_sup, friction_force, friction_loss = compute_BL_supersonic(
        r_t, eps, theta_in_deg, bell_fraction, p0=p0, T0=T0_val)
    n_BL_sub, ds_throat, L_conv = compute_BL_convergent(
        r_t, eps, theta_sub_deg, R_chamber, p0=p0, T0=T0_val)
    n_total   = n_div * n_BL_sup * n_BL_sub

    # ── Velocidade isentrópica ideal ──────────────────────────────────────────
    Me_ist  = mach_from_area_ratio(eps, supersonic=True)
    Ve_ist  = exit_velocity(Me_ist, T0=T0_val)

    # ── Resultados base ───────────────────────────────────────────────────────
    results = {
        # Inputs
        "r_t [m]"                 : r_t,
        "eps [-]"                 : eps,
        "theta_in [°]"            : theta_in_deg,
        "theta_out [°]"           : theta_out_calc_deg,
        "theta_sub [°]"           : theta_sub_deg,
        "bell_fraction [-]"       : bell_fraction,
        "R_chamber [m]"           : R_chamber,
        # Geometria divergente
        "r_e [m]"                 : r_e,
        "L [m]"                   : L,
        "theta_out_parabola [°]"  : theta_out_calc_deg,
        "a"                       : a,
        "b"                       : b,
        "c"                       : c,
        # Velocidade ideal
        "Me_ist [-]"              : Me_ist,
        "Ve_ist [m/s]"            : Ve_ist,
        # Camada limite convergente
        "delta_star_throat [m]"   : ds_throat,
        "r_t_eff [m]"             : r_t - ds_throat,
        "L_conv [m]"              : L_conv,
        # Rendimentos
        "eta_div"                 : n_div,
        "eta_BL_sup"              : n_BL_sup,
        "friction_force [N]"      : friction_force,
        "friction_loss [-]"       : friction_loss,
        "eta_BL_sub"              : n_BL_sub,
        "eta_total"               : n_total,
        # Referência sem BL convergente
        "eta_total_no_sub"        : n_div * n_BL_sup,
    }

    # ── Thrust / Isp (opcional — requer p0 e W) ───────────────────────────────
    if p0 is not None and W is not None:
        thrust_data = compute_thrust_isp(
            r_t       = r_t,
            eps       = eps,
            eta_total = n_total,
            Ve_ist    = Ve_ist,
            p0        = p0,
            T0_val    = T0_val,
            W         = W,
            p_amb     = p_amb,
            t_burn    = t_burn,
        )
        results.update(thrust_data)
        results["p0 [Pa]"]  = p0
        results["W [kg/mol]"] = W
        results["p_amb [Pa]"] = p_amb

    if verbose:
        _print_report(results)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RELATÓRIO
# ═══════════════════════════════════════════════════════════════════════════════

def _print_report(r):
    """Imprime relatório formatado."""
    W = 60
    sep  = "═" * W
    sep2 = "─" * W

    print(sep)
    print("  BELL NOZZLE EFFICIENCY — INVICTUS III / FEUP")
    print(sep)
    print(f"  Raio da garganta         r_t      = {r['r_t [m]']*1e3:.3f} mm")
    print(f"  Raio da câmara           R_ch     = {r['R_chamber [m]']*1e3:.2f} mm")
    print(f"  Coef. de expansão        ε        = {r['eps [-]']:.4f}")
    print(f"  Ângulo de entrada        θ_in     = {r['theta_in [°]']:.2f}°")
    print(f"  Ângulo de saída          θ_out    = {r['theta_out [°]']:.2f}°"
          f"  (parábola: {r['theta_out_parabola [°]']:.2f}°)")
    print(f"  Ângulo convergente       θ_sub    = {r['theta_sub [°]']:.2f}°")
    print(f"  Fração de bell           L/Lcone  = {r['bell_fraction [-]']:.3f}")
    print(sep2)
    print(f"  Raio de saída            r_e      = {r['r_e [m]']*1e3:.3f} mm")
    print(f"  Comprimento axial (sup.) L_div    = {r['L [m]']*1e3:.3f} mm")
    print(f"  Comprimento axial (conv.)L_conv   = {r['L_conv [m]']*1e3:.3f} mm")
    print(f"  Comprimento total        L_total  = {(r['L [m]']+r['L_conv [m]'])*1e3:.3f} mm")
    print(sep2)
    print("  CAMADA LIMITE CONVERGENTE")
    print(f"    δ* na garganta         δ*_t     = {r['delta_star_throat [m]']*1e6:.4f} μm")
    print(f"    Raio efectivo garganta r_t,eff  = {r['r_t_eff [m]']*1e3:.4f} mm"
          f"  (Δ = {(r['r_t [m]']-r['r_t_eff [m]'])*1e6:.3f} μm)")
    print(sep2)
    print("  RENDIMENTOS")
    print(f"    η_div   (divergência)  = {r['eta_div']:.6f}")
    print(f"    η_BL_sup (CL superson.)= {r['eta_BL_sup']:.6f}")
    print(f"    η_BL_sub (CL converg.) = {r['eta_BL_sub']:.6f}")
    print(f"    F_fric  (divergente)   = {r['friction_force [N]']:.4f} N")
    print(sep2)
    print(f"  η_TOTAL (com BL conv.)   = {r['eta_total']:.6f}  ({r['eta_total']*100:.4f} %)")
    print(f"  η_TOTAL (sem BL conv.)   = {r['eta_total_no_sub']:.6f}  ({r['eta_total_no_sub']*100:.4f} %)")
    delta_pct = (r['eta_total_no_sub'] - r['eta_total']) * 100
    print(f"  Δη (impacto BL conv.)    = {delta_pct:.4f} pp")

    # ── Thrust / Isp (só se calculados) ──────────────────────────────────────
    if "F [N]" in r:
        print(sep2)
        print("  THRUST & IMPULSO ESPECÍFICO  (INV-PROP-NC-XXX Rev. R1)")
        print(f"    p₀ câmara              = {r['p0 [Pa]']/1e5:.4f} bar")
        print(f"    W gases                = {r['W [kg/mol]']*1e3:.2f} g/mol")
        print(f"    p_amb                  = {r['p_amb [Pa]']/1e2:.2f} hPa")
        print(sep2)
        print(f"    ṁ  (débito mássico)    = {r['mdot [kg/s]']*1e3:.4f} g/s")
        print(f"    Me (Mach saída)        = {r['Me [-]']:.4f}")
        print(f"    Ve,ist (ideal)         = {r['Ve_ist [m/s]']:.2f} m/s")
        print(f"    Ve (corrigido)         = {r['Ve [m/s]']:.2f} m/s")
        print(f"    pe (pressão saída)     = {r['pe [Pa]']/1e2:.2f} hPa")
        print(sep2)
        print(f"    F_momentum             = {r['F_momentum [N]']:.4f} N")
        print(f"    F_pressure             = {r['F_pressure [N]']:.4f} N")
        print(f"    F  (thrust total)      = {r['F [N]']:.4f} N")
        print(f"    Isp                    = {r['Isp [s]']:.4f} s")
        if "J [N·s]" in r:
            print(f"    t_burn                 = {r['t_burn [s]']:.2f} s")
            print(f"    mp (massa propulsor)   = {r['mp [kg]']*1e3:.4f} g")
            print(f"    J  (impulso total)     = {r['J [N·s]']:.4f} N·s")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    compute_nozzle_efficiency(
        r_t=0.01531, eps=5.6, theta_in_deg=30.0, bell_fraction=0.80,
        theta_sub_deg=50.0, R_chamber=0.060,
        p0=50e5, T0_prop=3181.8, W=0.024064, p_amb=0.0,
        verbose=True,
    )
