#!/usr/bin/env python3
"""
Quasi-1D nozzle solver:
- Reads geometry (x, r) from CSV.
- Computes A(x), finds throat.
- Solves for M(x) from A/A_t via area–Mach relation.
- Computes T(x), P(x), v(x).
- Plots M(x), P(x), T(x).

Assumptions:
- Perfect gas, constant gamma.
- Isentropic, quasi-1D, choked throat.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# =========================
# ======= CONFIG ==========
# =========================

CSV_FILE = "nozzle_geometry.csv"   # <-- mete aqui o teu CSV exportado pelo NozzleGeometry

# Thermodynamic / gas model
Pc_bar = 35.0          # chamber / total pressure in bar (P0)
T0_K   = 3300.0        # total temperature [K] (mete aqui o T_c da CEA)
gamma  = 1.20          # effective gamma (do RocketCEA, ~1.15–1.25 típico)
R_gas  = 300.0         # J/(kg·K), effective gas constant (RocketCEA dá-te isto)

# Solver settings
M_sub_bracket = (1e-4, 0.999)   # bracket for subsonic solution
M_sup_bracket = (1.001, 10.0)   # bracket for supersonic solution
tol_M         = 1e-6            # tolerance in Mach
max_iter      = 100             # max iterations per point

# =========================
# === CORE FUNCTIONS ======
# =========================

def area_ratio_from_M(M, gamma):
    """
    A/A* as function of M and gamma (isentropic perfect-gas nozzle).
    """
    term = (2.0/(gamma+1.0)) * (1.0 + (gamma-1.0)/2.0 * M**2)
    return (1.0/M) * term**((gamma+1.0)/(2.0*(gamma-1.0)))


def F_M(M, RA_target, gamma):
    """
    Nonlinear function F(M; RA, gamma) = A(M)/A* - RA_target.
    Root F=0 -> Mach corresponding to area ratio RA_target.
    """
    return area_ratio_from_M(M, gamma) - RA_target


def bisection_solve_RA(RA_target, gamma, bracket, tol=1e-6, max_iter=100):
    """
    Simple bisection root-finder for F(M;RA,gamma)=0 within a given bracket.
    Assumes F changes sign over the interval.
    """
    a, b = bracket
    fa = F_M(a, RA_target, gamma)
    fb = F_M(b, RA_target, gamma)

    if fa * fb > 0.0:
        # No sign change: return None to signal failure
        return None

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = F_M(c, RA_target, gamma)

        if abs(fc) < tol or (b - a) < tol:
            return c

        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return 0.5 * (a + b)


def solve_M_distribution(x, r, gamma):
    """
    Given x and r(x) (in meters), compute:
    - A(x) = pi r^2
    - throat index, area ratio RA(x)
    - M(x) from RA(x) using bisection (subsonic before throat, supersonic after)

    Returns:
      x_sorted, A, M
    """

    # Sort by x (just in case CSV isn't sorted)
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    r = r[sort_idx]

    A = math.pi * r**2
    i_throat = np.argmin(A)
    At = A[i_throat]

    RA = A / At

    M = np.zeros_like(x)

    # At throat: M=1
    M[i_throat] = 1.0

    # Solve subsonic branch (i < i_throat)
    for i in range(i_throat):
        RA_i = RA[i]
        sol = bisection_solve_RA(RA_i, gamma, M_sub_bracket,
                                 tol=tol_M, max_iter=max_iter)
        if sol is None:
            # fallback: set near-sonic
            M[i] = 0.99
        else:
            M[i] = sol

    # Solve supersonic branch (i > i_throat)
    for i in range(i_throat + 1, len(x)):
        RA_i = RA[i]
        sol = bisection_solve_RA(RA_i, gamma, M_sup_bracket,
                                 tol=tol_M, max_iter=max_iter)
        if sol is None:
            # fallback: set slightly supersonic
            M[i] = 1.01
        else:
            M[i] = sol

    return x, A, M


def thermo_from_M(M, P0, T0, gamma, R_gas):
    """
    From Mach distribution M(x), total conditions (P0,T0) and gas parameters,
    compute static P(x), T(x), v(x).
    """
    # T/T0
    T_T0 = 1.0 / (1.0 + 0.5*(gamma-1.0)*M**2)
    T = T0 * T_T0

    # P/P0 = (T/T0)^{gamma/(gamma-1)}
    P_P0 = T_T0**(gamma/(gamma-1.0))
    P = P0 * P_P0

    # v = M * sqrt(gamma R T)
    v = M * np.sqrt(gamma * R_gas * T)

    return P, T, v


# =========================
# ========= MAIN ==========
# =========================

def main():
    # --- Load geometry from CSV ---
    # Expect first column x [m], second column r [m].
    # If o teu CSV tiver header, ajusta skiprows.
    x, r = np.loadtxt(CSV_FILE, delimiter=',', skiprows=1, usecols=(0,1), unpack=True)

    # --- Solve for M(x) ---
    x_sorted, A, M = solve_M_distribution(x, r, gamma)

    # --- Thermodynamics ---
    P0_Pa = Pc_bar * 1e5   # P0 = Pc (assumido isentropia câmara->garganta)
    P, T, v = thermo_from_M(M, P0_Pa, T0_K, gamma, R_gas)

    # --- Plots ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(x_sorted, M)
    axs[0].set_ylabel("Mach M(x)")
    axs[0].grid(True)

    axs[1].plot(x_sorted, P / 1e5)   # bar
    axs[1].set_ylabel("Pressure P(x) [bar]")
    axs[1].grid(True)

    axs[2].plot(x_sorted, T)
    axs[2].set_ylabel("Temperature T(x) [K]")
    axs[2].set_xlabel("Axial coordinate x [m]")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # If you want, also print some key values:
    i_throat = np.argmin(A)
    i_exit   = np.argmax(x_sorted)

    print(f"Throat at x = {x_sorted[i_throat]:.4f} m, M = {M[i_throat]:.3f}, "
          f"P = {P[i_throat]/1e5:.3f} bar, T = {T[i_throat]:.1f} K")

    print(f"Exit at x = {x_sorted[i_exit]:.4f} m, M = {M[i_exit]:.3f}, "
          f"P = {P[i_exit]/1e5:.3f} bar, T = {T[i_exit]:.1f} K")

if __name__ == "__main__":
    main()
