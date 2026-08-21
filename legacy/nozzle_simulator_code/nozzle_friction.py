# nozzle_friction.py
import numpy as np
import math
import re as re_mod

# ============================================================
# Robust Area–Mach relations (NO fsolve)
# ============================================================

def area_mach(M, gamma):
    return (1.0 / M) * (
        (2.0 / (gamma + 1.0)) *
        (1.0 + (gamma - 1.0) * 0.5 * M * M)
    ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area_ratio_bisect(A_over_Astar, gamma, supersonic):
    A = max(float(A_over_Astar), 1.0 + 1e-12)

    def f(M):
        return area_mach(M, gamma) - A

    if supersonic:
        lo, hi = 1.0 + 1e-8, 30.0
    else:
        lo, hi = 1e-6, 1.0 - 1e-8

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return 2.0 if supersonic else 0.2

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    return 0.5 * (lo + hi)


# ============================================================
# Fallback mu(T) (ONLY used if CEA parse fails)
# ============================================================

def mu_sutherland(T, mu_ref=3.0e-5, T_ref=300.0, S=110.4):
    T = max(float(T), 1.0)
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


# ============================================================
# Parse viscosity from CEA full text output
# ============================================================

def _parse_floats_from_line(line):
    return [float(x) for x in re_mod.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]


def cea_mu_triplet(C, Pc_bar, MR, eps):
    """
    Returns dict with mu at chamber/throat/exit from CEA output:
      {"Tc":..., "muc":..., "Tt":..., "mut":..., "Te":..., "mue":...}
    mu in Pa.s, T in K
    If parsing fails, returns None.
    """
    try:
        txt = C.get_full_cea_output(Pc=Pc_bar, MR=MR, eps=eps, frozen=0, frozenAtThroat=0)
    except Exception:
        return None

    lines = txt.splitlines()

    T_vals = None
    mu_mP_vals = None

    # temperature row
    for ln in lines:
        u = ln.upper().strip()
        if (("T," in u or u.startswith("T")) and ("K" in u or "DEGK" in u)) and ("VISC" not in u):
            nums = _parse_floats_from_line(ln)
            if len(nums) >= 3:
                T_vals = nums[-3:]
                break

    # viscosity row (milliPoise)
    for ln in lines:
        u = ln.upper().strip()
        if "VISC" in u and ("MILLIPOISE" in u or "MPOISE" in u or "M-POISE" in u):
            nums = _parse_floats_from_line(ln)
            if len(nums) >= 3:
                mu_mP_vals = nums[-3:]
                break

    # fallback: any VISC line with 3 numbers (assume mP)
    if mu_mP_vals is None:
        for ln in lines:
            u = ln.upper().strip()
            if u.startswith("VISC"):
                nums = _parse_floats_from_line(ln)
                if len(nums) >= 3:
                    mu_mP_vals = nums[-3:]
                    break

    if T_vals is None or mu_mP_vals is None:
        return None

    mu_vals = [m * 1e-4 for m in mu_mP_vals]  # mP -> Pa.s

    return {
        "Tc": float(T_vals[0]), "muc": float(mu_vals[0]),
        "Tt": float(T_vals[1]), "mut": float(mu_vals[1]),
        "Te": float(T_vals[2]), "mue": float(mu_vals[2]),
    }


def build_mu_of_T_from_cea(C, Pc_bar, MR, eps):
    """
    Build mu(T) interpolator from 3 CEA points (chamber, throat, exit).
    Uses log-log linear fit (power-law) for smoothness.
    Returns (mu_of_T, trip_dict_or_None).
    """
    trip = cea_mu_triplet(C, Pc_bar, MR, eps)
    if trip is None:
        def mu_of_T(T):
            return mu_sutherland(T)
        return mu_of_T, None

    T_pts = np.array([trip["Tc"], trip["Tt"], trip["Te"]], dtype=float)
    mu_pts = np.array([trip["muc"], trip["mut"], trip["mue"]], dtype=float)

    lnT = np.log(np.maximum(T_pts, 1.0))
    lnmu = np.log(np.maximum(mu_pts, 1e-12))
    b, a = np.polyfit(lnT, lnmu, 1)  # ln(mu)=a + b ln(T)

    def mu_of_T(T):
        T = max(float(T), 1.0)
        return float(np.exp(a + b * np.log(T)))

    return mu_of_T, trip


# ============================================================
# Main API
# ============================================================

def friction_bl_cea(
    x_all,
    r_all,
    rt,
    translation_x,
    Tc,
    Pc_bar,
    MR,
    exp,
    C,
    plot_debug=False
):
    """
    Boundary-layer blockage model with:
      - continuous delta*
      - viscosity mu(T) extracted from CEA output (3 points fit)
      - robust area->Mach inversion (bisection)

    Inputs:
      x_all, r_all: arrays defining nozzle wall radius vs x (must be same size)
      rt: throat radius [m]
      translation_x: throat x-position [m]
      Tc: chamber temperature [K] (stagnation approx)
      Pc_bar: chamber pressure [bar]
      MR: O/F
      exp: expansion ratio (Ae/At)
      C: rocketcea CEA_Obj instance
      plot_debug: if True, returns extra arrays (no plotting here)

    Returns dict:
      eta_v, Ma_iso, Ma_eff, delta_star_m, r_eff_m, mu_Pa_s, Re_x_plot, cea_mu_triplet
    """

    x_all = np.asarray(x_all, dtype=float).copy()
    r_all = np.asarray(r_all, dtype=float).copy()

    mask = np.isfinite(x_all) & np.isfinite(r_all)
    x_all = x_all[mask]
    r_all = r_all[mask]

    idx = np.argsort(x_all)
    x_all = x_all[idx]
    r_all = r_all[idx]

    # Gas constants (gamma, MW) from CEA at chamber conditions
    MW_g_per_mol, gamma_local = C.get_Chamber_MolWt_gamma(Pc=Pc_bar, MR=MR)
    MW = MW_g_per_mol / 1000.0
    Ru = 8.314462618
    R = Ru / MW

    T0 = float(Tc)
    P0 = float(Pc_bar) * 1e5  # Pa

    # mu(T) from CEA (3-pt fit)
    mu_of_T, mu_trip = build_mu_of_T_from_cea(C, Pc_bar, MR, exp)

    # Geometry -> A/A*
    At = math.pi * rt**2
    A_geom = math.pi * r_all**2
    A_over_Astar = A_geom / At

    # Ideal Mach from geometry
    Ma_iso = np.array([
        mach_from_area_ratio_bisect(AoA, gamma_local, supersonic=(x >= translation_x))
        for x, AoA in zip(x_all, A_over_Astar)
    ], dtype=float)

    # BL model
    delta_star = np.zeros_like(x_all, dtype=float)
    r_eff = np.zeros_like(r_all, dtype=float)
    mu_all = np.zeros_like(x_all, dtype=float)
    Rex_plot = np.zeros_like(x_all, dtype=float)

    x_start = float(x_all[0])
    x_throat = float(translation_x)
    delta_throat = None

    for i, (x, r, M) in enumerate(zip(x_all, r_all, Ma_iso)):

        # continuous x for Re plotting
        x_plot = max(x - x_start, 1e-4)

        # growth x: convergente conta desde start; divergente conta desde throat
        x_growth = max((x - x_throat) if (x >= x_throat) else (x - x_start), 1e-4)

        # local isentropic props
        T = T0 / (1.0 + (gamma_local - 1.0) * 0.5 * M**2)
        P = P0 / (1.0 + (gamma_local - 1.0) * 0.5 * M**2) ** (gamma_local / (gamma_local - 1.0))

        rho = P / (R * T)
        a = math.sqrt(gamma_local * R * T)
        U = M * a

        mu_loc = float(mu_of_T(T))
        mu_all[i] = mu_loc

        Rex_p = max(rho * U * x_plot / mu_loc, 1e3)
        Rex_g = max(rho * U * x_growth / mu_loc, 1e3)
        Rex_plot[i] = Rex_p

        # turbulent displacement thickness growth
        delta_growth = 0.046 * x_growth / (Rex_g ** 0.2)

        if x < x_throat:
            delta = delta_growth
        else:
            if delta_throat is None:
                delta_throat = float(delta_star[i-1]) if i > 0 else 0.0
            delta = delta_throat + delta_growth  # continuous at throat

        delta_star[i] = delta
        r_eff[i] = max(r - delta, 1e-6)

    # Effective Mach with reduced area
    A_eff = math.pi * r_eff**2
    Aeff_over_Astar = A_eff / At

    Ma_eff = np.array([
        mach_from_area_ratio_bisect(AoA, gamma_local, supersonic=(x >= translation_x))
        for x, AoA in zip(x_all, Aeff_over_Astar)
    ], dtype=float)

    # Exit velocity efficiency
    Me_iso = Ma_iso[-1]
    Te_iso = T0 / (1.0 + (gamma_local - 1.0) * 0.5 * Me_iso**2)
    Ve_iso = Me_iso * math.sqrt(gamma_local * R * Te_iso)

    Me_eff = Ma_eff[-1]
    Te_eff = T0 / (1.0 + (gamma_local - 1.0) * 0.5 * Me_eff**2)
    Ve_eff = Me_eff * math.sqrt(gamma_local * R * Te_eff)

    eta_v = float(Ve_eff / Ve_iso)

    out = {
        "eta_v": eta_v,
        "x_m": x_all,
        "r_m": r_all,
        "Ma_iso": Ma_iso,
        "Ma_eff": Ma_eff,
        "delta_star_m": delta_star,
        "r_eff_m": r_eff,
        "mu_Pa_s": mu_all,
        "Re_x_plot": Rex_plot,
        "cea_mu_triplet": mu_trip,  # None if fallback
    }

    # Optional debug: do NOT plot here, just return arrays (main can plot)
    if plot_debug:
        out["gamma"] = float(gamma_local)
        out["MW_kg_per_mol"] = float(MW)

    return out
