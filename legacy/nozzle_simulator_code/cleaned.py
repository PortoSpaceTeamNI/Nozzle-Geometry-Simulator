import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

from nozzle_friction import friction_bl_cea
from k import *

import re

# ============================================================
# RocketCEA setup
# ============================================================
card_str = """
fuel C30H62 C 30 H 62 wt%=83.0
h,cal=-191921.61  t(k)=298.15
fuel C8H8 C 8 H 8 wt%=7.004
h,cal=35444.55  t(k)=298.15
fuel C4H6 C 4 H 6 wt%=4.998
h,cal=26290.63  t(k)=298.15
fuel C3H3N C 3 H 3 N 1 wt%=4.998
h,cal=35156.79  t(k)=298.15
"""
add_new_fuel("Paraffin", card_str)
C = CEA_Obj(propName="", oxName="N2O", fuelName="Paraffin")

# ============================================================
# Helpers: robust float parsing from CEA full output
# ============================================================
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

def _nums(line: str):
    return [float(x) for x in _FLOAT_RE.findall(line)]

def get_gamma_triplet(Pc_bar, OF, eps, debug=False):
    """
    Returns (gamma_c, gamma_t, gamma_e) from CEA full output.
    Input Pc in bar (pc_units='bar').
    """
    s = C.get_full_cea_output(Pc=Pc_bar, MR=OF, eps=eps, output="siunits", pc_units="bar")
    if debug:
        print(s)
    for line in s.splitlines():
        u = line.strip().upper()
        if u.startswith("GAMMA"):
            vals = _nums(line)
            # Typical: GAMMA  <chamber> <throat> <exit>
            if len(vals) >= 3:
                return float(vals[0]), float(vals[1]), float(vals[2])
    raise RuntimeError("Could not parse GAMMA line from CEA full output.")

def get_gas_transport_props(Pc_bar, OF, eps,
                            col=1,  # 0 chamber, 1 throat, 2 exit
                            default_mu=3.5e-5, default_Cp=3000.0, default_Pr=0.7, default_cstar=1500.0,
                            debug=False):
    """
    Parse mu, Cp, Pr, c* from CEA full output.
    Input Pc in bar (pc_units='bar').

    Returns:
      mu    [Pa.s]
      Cp    [J/kg/K]
      Pr    [-]
      cstar [m/s]
    """
    mu = float(default_mu)
    Cp = float(default_Cp)
    Pr = float(default_Pr)
    cstar = float(default_cstar)

    s = C.get_full_cea_output(Pc=Pc_bar, MR=OF, eps=eps, output="siunits", pc_units="bar", short_output=0)
    if debug:
        print(s)

    for line in s.splitlines():
        u = line.strip().upper()

        # c*
        if ("CSTAR" in u) and ("M/SEC" in u or "M/S" in u or "M-SEC" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                cstar = vals[col] if col < len(vals) else vals[-1]

        # viscosity
        if ("VISC" in u) or ("VISCOSITY" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                mu_val = vals[col] if col < len(vals) else vals[-1]

                # defensive unit conversions
                if "MICROPOISE" in u:
                    mu_val *= 1e-7
                elif "MILLIPOISE" in u:
                    mu_val *= 1e-4
                elif "POISE" in u and "PA" not in u:
                    mu_val *= 0.1

                mu = float(mu_val)

        # Cp
        # Typical line: "Cp, KJ/(KG)(K)   <c> <t> <e>" or in J/(KG)(K)
        if (u.startswith("CP") or " CP" in u) and ("KG" in u) and ("K" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                cp_val = vals[col] if col < len(vals) else vals[-1]
                if "KJ" in u:
                    cp_val *= 1000.0
                Cp = float(cp_val)

        # Prandtl
        if "PRANDTL" in u:
            vals = _nums(line)
            if len(vals) >= 3:
                Pr = float(vals[col] if col < len(vals) else vals[-1])

    return mu, Cp, Pr, cstar

# ============================================================
# GUI globals
# ============================================================
root = tk.Tk()
root.title("Nozzle Profile Generator")

last_results = {}

# ============================================================
# Plot helper for friction model
# ============================================================
def plot_friction_results(res, translation_x, rt):
    x = res["x_m"]
    r = res["r_m"]
    Ma_iso = res["Ma_iso"]
    Ma_eff = res["Ma_eff"]
    delta = res["delta_star_m"]
    r_eff = res["r_eff_m"]
    mu = res["mu_Pa_s"]
    Rex = res["Re_x_plot"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, Ma_iso, label="Mach (geom)")
    plt.plot(x, Ma_eff, label="Mach (with BL)")
    plt.axvline(translation_x, linestyle="--", label="throat")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("Mach")
    plt.title("Mach profile — ideal vs BL-corrected")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, delta * 1e3)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("δ* (mm)")
    plt.title("Boundary-layer displacement thickness (continuous)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, r * 1e3, label="geometry")
    plt.plot(x, r_eff * 1e3, label="effective")
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("radius (mm)")
    plt.title("Radius reduction due to BL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    eps_geom = (r[-1] / rt) ** 2
    eps_eff  = (r_eff[-1] / rt) ** 2
    plt.figure(figsize=(10, 6))
    plt.plot(x, r_eff / r)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("r_eff / r")
    plt.title(f"Effective area loss (ε: geom={eps_geom:.3f}, eff={eps_eff:.3f})")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, mu)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("mu (Pa·s)")
    title = "Viscosity mu(T) from CEA (fit)" if (res.get("cea_mu_triplet") is not None) else "Viscosity mu(T) fallback"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, Rex)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("Re_x")
    plt.title("Re_x along nozzle (continuous x)")
    plt.tight_layout()
    plt.show()

    trip = res.get("cea_mu_triplet")
    if trip is not None:
        print("CEA mu triplet (T[K], mu[Pa.s]):",
              (trip["Tc"], trip["muc"]),
              (trip["Tt"], trip["mut"]),
              (trip["Te"], trip["mue"]))

# ============================================================
# Main simulation
# ============================================================
def run_simulation(mode):
    global last_results

    try:
        # ---------------------------
        # User Inputs
        # ---------------------------
        Tc = float(entry_Tc.get())
        Pc = float(entry_Pc.get())          # bar
        MR = float(entry_MR.get())          # O/F
        rt = float(entry_rt.get())          # m
        exp = float(entry_exp.get())        # epsilon Ae/At
        halfangle = math.radians(float(entry_halfangle.get()))
        parabola_multiplier = float(entry_parabmultiplier.get())
        theta_in = math.radians(float(entry_theta_in.get()))
        theta_sub = math.radians(float(entry_theta_sub.get()))
        R_chamber = float(entry_Rchamber.get()) / 2.0
        bell_contour = float(entry_bell_contour.get()) / 100.0
        cr = (R_chamber**2) / (rt**2)

        # ---------------------------
        # Subsonic arc intersection
        # ---------------------------
        xr = -1.5 * rt * math.sin(theta_sub)
        yr = 2.5 * rt - 1.5 * rt * math.cos(theta_sub)

        if R_chamber < yr:
            warning_label.config(text="Chamber Diameter too Small for Selected Angle!", fg="red")
            return

        m_line = -math.tan(theta_sub)
        b_line = yr - m_line * xr
        xf = (R_chamber - b_line) / m_line

        def line(x):
            return m_line * x + b_line

        # ---------------------------
        # IMPORTANT: translation_x must be computed early (used everywhere)
        # ---------------------------
        translation_x = -xr + (R_chamber - yr) / math.tan(theta_sub)

        # ---------------------------
        # Supersonic arc and parabola
        # ---------------------------
        re_exit = math.sqrt(exp) * rt
        Lcone = (re_exit - rt) / math.tan(halfangle)
        Lparab = bell_contour * Lcone * parabola_multiplier

        r_sup = 0.4 * rt
        yc_sup = 1.4 * rt

        angle_vals = np.linspace(math.pi / 2, math.pi / 2 - theta_in, 100)
        arc_x0 = r_sup * np.cos(angle_vals)           # before translation
        arc_y  = yc_sup - r_sup * np.sin(angle_vals)

        Px = float(arc_x0[-1])
        Py = float(arc_y[-1])
        m0 = math.tan(theta_in)
        Lfinal0 = Px + Lparab                          # before translation

        # parabola coefficients: y = a x^2 + b x + c  (in pre-translation x)
        A = np.array([
            [Px**2, Px, 1],
            [2*Px, 1, 0],
            [Lfinal0**2, Lfinal0, 1]
        ])
        Y = np.array([Py, m0, re_exit])
        a, b_parab, c_parab = np.linalg.solve(A, Y)

        if a > 0:
            warning_label.config(text="Inverted Parabola (a > 0)", fg="red")
            return
        else:
            warning_label.config(text="")

        theta_out_rad = math.atan(2*a*Lfinal0 + b_parab)
        theta_out_deg = abs(math.degrees(theta_out_rad))

        # build parabola samples (pre-translation)
        x_vals0 = np.linspace(Px, Lfinal0, 300)
        y_vals  = a * x_vals0**2 + b_parab * x_vals0 + c_parab

        if np.any(y_vals > re_exit + 1e-4):
            warning_label.config(
                text="Parabola Exceeds Exit Radius!\n"
                     f"Max y Value: {np.max(y_vals)}\n"
                     f"Exit Radius: {re_exit}",
                fg="red"
            )
            return

        # ---------------------------
        # Apply translation to x arrays
        # ---------------------------
        def correct_translation(x_vals_list, tx):
            x_out = [float(x + tx) for x in x_vals_list]
            if len(x_out) > 0 and abs(x_out[0]) < 1e-6:
                x_out[0] = 0.0
            return x_out

        xs0_0 = np.linspace(xf, xr, 100)  # pre-translation
        ys0   = line(xs0_0)

        # Smoothed subsonic arc (pre-translation)
        def arc_sub(pre_x):
            val = 2.25 * rt**2 - pre_x**2
            if val < 0:
                return None
            return 2.5 * rt - math.sqrt(max(0.0, val))

        xs1_0, ys1 = [], []
        pre_x = 0.0
        while pre_x > xr:
            val = arc_sub(pre_x)
            if val is not None:
                xs1_0.append(pre_x)
                ys1.append(val)
            pre_x -= 0.0001

        # translate all x arrays
        xs0 = np.array(correct_translation(xs0_0, translation_x), dtype=float)
        xs1 = np.array(correct_translation(xs1_0, translation_x), dtype=float)
        arc_x = np.array(correct_translation(arc_x0, translation_x), dtype=float)
        x_vals = np.array(correct_translation(x_vals0, translation_x), dtype=float)

        ys0 = np.array(ys0, dtype=float)
        ys1 = np.array(ys1, dtype=float)
        arc_y = np.array(arc_y, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        # exit x (translated)
        x_exit = float(Lfinal0 + translation_x)

        # ---------------------------
        # Build full stations (for friction)
        # ---------------------------
        x_all = np.concatenate([xs0, xs1, arc_x, x_vals]).astype(float)
        r_all = np.concatenate([ys0, ys1, arc_y, y_vals]).astype(float)

        mask = np.isfinite(x_all) & np.isfinite(r_all)
        x_all = x_all[mask]
        r_all = r_all[mask]
        idx = np.argsort(x_all)
        x_all = x_all[idx]
        r_all = r_all[idx]

        # ---------------------------
        # Geometry query y(x) (translated x)
        # ---------------------------
        def find_y_from_x(x):
            x = float(x)

            # supersonic parabola
            if x >= (Px + translation_x):
                xt = x - translation_x
                return float(a * xt**2 + b_parab * xt + c_parab)

            # supersonic arc
            if (x >= translation_x) and (x < (Px + translation_x)):
                dx = x - translation_x
                inside = (0.4*rt)**2 - dx**2
                inside = max(0.0, inside)
                return float(1.4*rt - math.sqrt(inside))

            # subsonic arc
            if (x >= (xr + translation_x)) and (x < translation_x):
                dx = x - translation_x
                inside = 2.25*rt**2 - dx**2
                inside = max(0.0, inside)
                return float(2.5*rt - math.sqrt(inside))

            # subsonic line
            dx = x - translation_x
            return float(m_line*dx + b_line)

        # ---------------------------
        # CEA gamma triplet + gamma(x) smoothing
        # ---------------------------
        gamma_c, gamma_t, gamma_e = get_gamma_triplet(Pc, MR, exp)

        def clamp01(s):
            return 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

        def smoothstep(s):
            return s*s*(3.0 - 2.0*s)

        def gamma_x(x):
            # smooth ramp throat->exit in diverging section based on eps(x)=A/At
            x = float(x)
            if x <= translation_x:
                return float(gamma_t)
            r = find_y_from_x(x)
            eps_x = (r/rt)**2
            s = (eps_x - 1.0)/(exp - 1.0)
            s = smoothstep(clamp01(s))
            return float(gamma_t + (gamma_e - gamma_t)*s)

        # IMPORTANT: Area–Mach inversion assumes constant gamma.
        # Keep it stable: use gamma_t in subsonic, gamma_e in supersonic.
        def gamma_for_mach(x):
            return float(gamma_t if x <= translation_x else gamma_e)

        # ---------------------------
        # Mach / T / P
        # ---------------------------
        def area_mach(Ma, gam):
            return (1.0/Ma) * ( (2.0/(gam+1.0))*(1.0 + (gam-1.0)/2.0*Ma**2) )**((gam+1.0)/(2.0*(gam-1.0)))

        def find_mach(x):
            x = float(x)
            y = find_y_from_x(x)
            target = (y / rt)**2  # A/At = (r/rt)^2

            gam = gamma_for_mach(x)

            def func(Ma):
                Ma = float(Ma)
                return area_mach(Ma, gam) - target

            Ma_guess = 2.0 if x >= translation_x else 0.2
            sol = float(fsolve(func, Ma_guess)[0])

            # sanity clamp
            if sol <= 0:
                sol = 1e-6
            return sol

        def find_temp(x, Ma):
            gam = gamma_x(x)
            return float(Tc / (1.0 + (gam - 1.0)/2.0 * Ma**2))

        def find_pres(x, Ma):
            gam = gamma_x(x)
            return float(Pc * (1.0/(1.0 + (gam - 1.0)/2.0 * Ma**2))**(gam/(gam - 1.0)))

        # ---------------------------
        # Arrays: THIS is what you were missing (zip x with Ma)
        # ---------------------------
        Mas0 = np.array([find_mach(x) for x in xs0], dtype=float)
        Tas0 = np.array([find_temp(x, Ma) for x, Ma in zip(xs0, Mas0)], dtype=float)
        Pas0 = np.array([find_pres(x, Ma) for x, Ma in zip(xs0, Mas0)], dtype=float)

        Mas1 = np.array([find_mach(x) for x in xs1], dtype=float)
        Tas1 = np.array([find_temp(x, Ma) for x, Ma in zip(xs1, Mas1)], dtype=float)
        Pas1 = np.array([find_pres(x, Ma) for x, Ma in zip(xs1, Mas1)], dtype=float)

        Ma_arc = np.array([find_mach(x) for x in arc_x], dtype=float)
        Ta_arc = np.array([find_temp(x, Ma) for x, Ma in zip(arc_x, Ma_arc)], dtype=float)
        Pa_arc = np.array([find_pres(x, Ma) for x, Ma in zip(arc_x, Ma_arc)], dtype=float)

        Ma_vals = np.array([find_mach(x) for x in x_vals], dtype=float)
        Ta_vals = np.array([find_temp(x, Ma) for x, Ma in zip(x_vals, Ma_vals)], dtype=float)
        Pa_vals = np.array([find_pres(x, Ma) for x, Ma in zip(x_vals, Ma_vals)], dtype=float)

        # throat conditions
        P_star = find_pres(translation_x, 1.0)
        T_star = find_temp(translation_x, 1.0)

        # ---------------------------
        # Plot functions
        # ---------------------------
        def plot2d():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, ys0, "r--", label="Initial Straight Line")
            plt.plot(xs1, ys1, "b", label="Subsonic Arc")
            plt.plot(arc_x, arc_y, "g", label="Supersonic Arc")
            plt.plot(x_vals, y_vals, "k", label="Parabolic Contour")
            plt.legend()
            plt.grid()
            plt.xlabel("x (m)")
            plt.ylabel("radius y (m)")
            length = x_exit - float(xs0[0])
            plt.title("2D Nozzle Profile", pad=30)
            plt.figtext(0.1, 0.94, f"Contraction Ratio  = {cr:.3f}", fontsize=10, ha="left")
            plt.figtext(0.1, 0.91, f"Expansion Ratio = {exp:.3f}", fontsize=10, ha="left")
            plt.figtext(0.7, 0.94, f"Nozzle Length = {length:.3f} m", fontsize=10, ha="left")
            plt.figtext(0.7, 0.91, f"Parabola Exit Angle = {theta_out_deg:.3f}$^\\circ$", fontsize=10, ha="left")
            plt.gca().set_aspect("equal")
            plt.tight_layout()
            plt.show()

        def plot_Mach():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Mas0, "r--", label="Initial Straight Line")
            plt.plot(xs1, Mas1, "b", label="Subsonic Arc")
            plt.plot(arc_x, Ma_arc, "g", label="Supersonic Arc")
            plt.plot(x_vals, Ma_vals, "k", label="Parabolic Contour")
            plt.title("Mach Curve", pad=30)
            plt.figtext(0.8, 0.94, f"Starting Mach = {find_mach(float(xs0[0])):.3f}", fontsize=10, ha="left")
            plt.figtext(0.8, 0.91, f"Exit Mach = {Ma_vals[-1]:.3f}", fontsize=10, ha="left")
            plt.figtext(0.08, 0.94, f"Choking Point = {translation_x:.3f} m", fontsize=10, ha="left")
            plt.xlabel("x (m)")
            plt.ylabel("Ma")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        def plot_Temp():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Tas0, "r--", label="Initial Straight Line")
            plt.plot(xs1, Tas1, "b", label="Subsonic Arc")
            plt.plot(arc_x, Ta_arc, "g", label="Supersonic Arc")
            plt.plot(x_vals, Ta_vals, "k", label="Parabolic Contour")
            plt.title("Temperature Profile", pad=30)
            plt.figtext(0.68, 0.94, f"Main Chamber Temperature = {Tc:.3f} K", fontsize=10, ha="left")
            plt.figtext(0.68, 0.91, f"Exit Temperature = {Ta_vals[-1]:.3f} K", fontsize=10, ha="left")
            plt.figtext(0.08, 0.94, f"T at throat (M=1) = {T_star:.3f} K", fontsize=10, ha="left")
            plt.xlabel("x (m)")
            plt.ylabel("T (K)")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        def plot_Pres():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Pas0, "r--", label="Initial Straight Line")
            plt.plot(xs1, Pas1, "b", label="Subsonic Arc")
            plt.plot(arc_x, Pa_arc, "g", label="Supersonic Arc")
            plt.plot(x_vals, Pa_vals, "k", label="Parabolic Contour")
            plt.title("Pressure Profile", pad=30)
            plt.figtext(0.7, 0.94, f"Main Chamber Pressure = {Pc:.3f} bar", fontsize=10, ha="left")
            plt.figtext(0.7, 0.91, f"Exit Pressure = {Pa_vals[-1]:.3f} bar", fontsize=10, ha="left")
            plt.figtext(0.08, 0.94, f"Pressure at throat (M=1) = {P_star:.3f} bar", fontsize=10, ha="left")
            plt.xlabel("x (m)")
            plt.ylabel("P (bar)")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ---------------------------
        # CSV export (corrected x usage)
        # ---------------------------
        def update_csv():
            x_combined = [xs1, xs0, arc_x, x_vals]
            y_combined = [ys1, ys0, arc_y, y_vals]
            Ma_combined = [Mas1, Mas0, Ma_arc, Ma_vals]
            Ta_combined = [Tas1, Tas0, Ta_arc, Ta_vals]
            Pa_combined = [Pas1, Pas0, Pa_arc, Pa_vals]

            total_points = 100
            lengths = [len(arr) for arr in x_combined]
            total_len = sum(lengths)

            x_raw, y_raw, Ma_raw, Ta_raw, Pa_raw = [], [], [], [], []

            for x_seg, y_seg, Ma_seg, Ta_seg, Pa_seg, seg_len in zip(
                x_combined, y_combined, Ma_combined, Ta_combined, Pa_combined, lengths
            ):
                n_points = max(2, int((seg_len / total_len) * total_points))
                indices = np.linspace(0, seg_len - 1, n_points).astype(int)
                x_raw.extend([float(x_seg[i]) for i in indices])
                y_raw.extend([float(y_seg[i]) for i in indices])
                Ma_raw.extend([float(Ma_seg[i]) for i in indices])
                Ta_raw.extend([float(Ta_seg[i]) for i in indices])
                Pa_raw.extend([float(Pa_seg[i]) for i in indices])

            min_dist_mm = 1.0

            x_filtered = [x_raw[0]]
            y_filtered = [y_raw[0]]
            Ma_filtered = [Ma_raw[0]]
            Ta_filtered = [Ta_raw[0]]
            Pa_filtered = [Pa_raw[0]]

            last_x = x_raw[0]
            last_y = y_raw[0]
            last_Ma = Ma_raw[0]
            last_Ta = Ta_raw[0]
            last_Pa = Pa_raw[0]

            for xi, yi, Mai, Tai, Pai in zip(x_raw[1:], y_raw[1:], Ma_raw[1:], Ta_raw[1:], Pa_raw[1:]):
                dist = math.hypot(
                    (xi - last_x) * 1000.0,
                    (yi - last_y) * 1000.0,
                    (Mai - last_Ma) * 1000.0,
                    (Tai - last_Ta) * 1e-3,   # scale temp
                    (Pai - last_Pa) * 1000.0
                )
                if dist >= min_dist_mm:
                    x_filtered.append(xi)
                    y_filtered.append(yi)
                    Ma_filtered.append(Mai)
                    Ta_filtered.append(Tai)
                    Pa_filtered.append(Pai)
                    last_x, last_y, last_Ma, last_Ta, last_Pa = xi, yi, Mai, Tai, Pai

            x_final = x_filtered
            y_final = y_filtered
            Ma_final = Ma_filtered
            Ta_final = Ta_filtered
            Pa_final = Pa_filtered

            # force last point to be the true exit
            x_final[-1] = float(x_exit)
            y_final[-1] = float(re_exit)
            Ma_final[-1] = float(find_mach(x_exit))
            Ta_final[-1] = float(find_temp(x_exit, Ma_final[-1]))
            Pa_final[-1] = float(find_pres(x_exit, Ma_final[-1]))

            # meters -> mm
            x_final_mm = [x * 1000.0 for x in x_final]
            y_final_mm = [y * 1000.0 for y in y_final]

            combined = list(zip(x_final_mm, y_final_mm, Ma_final, Ta_final, Pa_final))
            combined.sort(key=lambda row: row[0])
            x_mm, y_mm, Ma_s, Ta_s, Pa_s = zip(*combined)

            try:
                with open("nozzle_geometry.csv", "w", encoding="utf-8") as f:
                    f.write("x,y,z\n")
                    for x, y in zip(x_mm, y_mm):
                        f.write(f"{x:.2f},{y:.2f},0\n")
                lbl_result.config(text="CSV Updated Successfully!", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Erro: {str(e)}", foreground="red")

            try:
                with open("mach_temp_pres_profile.csv", "w", encoding="utf-8") as f:
                    f.write("x_mm,Ma,T_K,P_bar\n")
                    for x, Ma, T, P in zip(x_mm, Ma_s, Ta_s, Pa_s):
                        f.write(f"{x:.2f},{Ma:.6f},{T:.6f},{P:.8f}\n")
                lbl_result.config(text="CSV Updated Successfully!", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Erro: {str(e)}", foreground="red")

        # ---------------------------
        # Theta out (correct)
        # ---------------------------
        def compute_theta_out_deg(a_, b_parab_, x_exit_pre_translation):
            return -math.degrees(math.atan(2.0 * a_ * x_exit_pre_translation + b_parab_))

        theta_out_deg_fixed = float(compute_theta_out_deg(a, b_parab, Lfinal0))

        # store last_results (do NOT reassign a new dict here)
        last_results.clear()
        last_results["theta_out_deg"] = theta_out_deg_fixed
        last_results["a"] = float(a)
        last_results["b_parab"] = float(b_parab)
        last_results["c"] = float(c_parab)
        last_results["Lfinal_pre"] = float(Lfinal0)
        last_results["translation_x"] = float(translation_x)
        last_results["gamma_c"] = float(gamma_c)
        last_results["gamma_t"] = float(gamma_t)
        last_results["gamma_e"] = float(gamma_e)
        last_results["Pe_bar"] = float(Pa_vals[-1])

        # ---------------------------
        # Thermal (Bartz simplified) — corrected gamma_local array
        # ---------------------------
        def run_thermal():
            Twg = 800.0
            r_recovery = 0.9

            Dt = 2.0 * rt
            At = math.pi * rt**2
            R_throat = 0.4 * rt

            mu, Cp, Pr, cstar = get_gas_transport_props(Pc, MR, exp, col=1)

            # stations along nozzle
            x_all_loc = x_all.copy()
            y_all_loc = r_all.copy()

            Ma_all = np.array([find_mach(float(xi)) for xi in x_all_loc], dtype=float)
            gam_all = np.array([gamma_x(float(xi)) for xi in x_all_loc], dtype=float)

            Taw = Tc * (1.0 + r_recovery * ((gam_all - 1.0) / 2.0) * Ma_all**2) / (1.0 + ((gam_all - 1.0) / 2.0) * Ma_all**2)

            sigma = (
                (0.5 * (Twg / Tc) * (1.0 + ((gam_all - 1.0) / 2.0) * Ma_all**2) + 0.5) ** (-0.68)
                * (1.0 + ((gam_all - 1.0) / 2.0) * Ma_all**2) ** (-0.12)
            )

            Pc_Pa = Pc * 1e5
            Ax = math.pi * y_all_loc**2

            const_bracket = (
                0.026 / (Dt**0.2)
                * ((mu**0.2) * Cp / (Pr**0.6))
                * ((Pc_Pa / cstar) ** 0.8)
                * ((Dt / R_throat) ** 0.1)
            )
            hg = const_bracket * ((At / Ax) ** 0.9) * sigma
            qdot = hg * (Taw - Twg)

            last_results["thermal"] = {
                "x_m": x_all_loc.copy(),
                "y_m": y_all_loc.copy(),
                "Ma": Ma_all.copy(),
                "gamma": gam_all.copy(),
                "Taw_K": Taw.copy(),
                "Twg_K": float(Twg),
                "sigma": sigma.copy(),
                "hg_W_m2K": hg.copy(),
                "qdot_W_m2": qdot.copy(),
                "mu_Pa_s": float(mu),
                "Cp_J_kgK": float(Cp),
                "Pr": float(Pr),
                "cstar_m_s": float(cstar),
            }

            plt.figure(figsize=(10, 6))
            plt.plot(x_all_loc, hg)
            plt.grid()
            plt.xlabel("x (m)")
            plt.ylabel("h_g (W/m²/K)")
            plt.title("Convective heat transfer coefficient (Bartz simplified)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(x_all_loc, qdot)
            plt.grid()
            plt.xlabel("x (m)")
            plt.ylabel("q'' (W/m²)")
            plt.title("Convective heat flux")
            plt.tight_layout()
            plt.show()

            try:
                with open("thermal_profile.csv", "w", encoding="utf-8") as f:
                    f.write("x_m,y_m,Ma,gamma,Taw_K,Twg_K,sigma,hg_W_m2K,qdot_W_m2\n")
                    for xi, yi, Mai, gi, Tawi, si, hgi, qi in zip(x_all_loc, y_all_loc, Ma_all, gam_all, Taw, sigma, hg, qdot):
                        f.write(f"{xi:.6f},{yi:.6f},{Mai:.6f},{gi:.6f},{Tawi:.3f},{Twg:.3f},{si:.6f},{hgi:.3f},{qi:.3f}\n")
                lbl_result.config(text="Thermal profile written to thermal_profile.csv", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Thermal CSV error: {e}", foreground="red")

            return last_results["thermal"]

        # ---------------------------
        # Modes
        # ---------------------------
        if mode == "2d":
            plot2d()
            return

        if mode == "Mach":
            plot_Mach()
            return

        if mode == "Temp":
            plot_Temp()
            return

        if mode == "Pres":
            plot_Pres()
            return

        if mode == "csv":
            update_csv()
            return

        if mode == "theta":
            return theta_out_deg_fixed

        if mode == "thermal":
            return run_thermal()

        if mode == "friction_bl_cea":
            res = friction_bl_cea(
                x_all=x_all,
                r_all=r_all,
                rt=rt,
                translation_x=translation_x,
                Tc=Tc,
                Pc_bar=Pc,
                MR=MR,
                exp=exp,
                C=C,
                plot_debug=True
            )
            last_results["friction_bl_cea"] = res
            eta = res["eta_v"]
            lbl_result.config(text=f"Friction (BL+CEA μ): η_v = {eta:.4f}", foreground="chartreuse4")
            plot_friction_results(res, translation_x, rt)
            return res

        if mode == "calibrate_k":
            Phi_ref, dbg = turning_metric_phi(x_all, r_all, translation_x, only_diverging=True)
            eta_turn_ref = 0.99
            k_cal = calibrate_k_from_reference(Phi_ref, eta_turn_ref=eta_turn_ref)
            last_results["Phi_ref"] = Phi_ref
            last_results["k_turn"] = k_cal
            last_results["eta_turn_ref"] = eta_turn_ref
            lbl_result.config(
                text=f"Calibrated k={k_cal:.4f} (Phi={Phi_ref:.4f} rad, eta_ref={eta_turn_ref:.3f})",
                foreground="chartreuse4"
            )
            return k_cal

        if mode == "score":
            k_turn = last_results.get("k_turn", 0.1)
            Phi, _ = turning_metric_phi(x_all, r_all, translation_x, only_diverging=True)
            eta_turn = eta_turn_from_phi(Phi, k_turn)

            eta_div = math.cos(math.radians(abs(theta_out_deg_fixed)))

            fr = friction_bl_cea(
                x_all=x_all, r_all=r_all, rt=rt, translation_x=translation_x,
                Tc=Tc, Pc_bar=Pc, MR=MR, exp=exp, C=C, plot_debug=False
            )
            eta_fric = fr["eta_v"]
            eta_total = eta_div * eta_fric * eta_turn

            last_results["eta_turn"] = eta_turn
            last_results["eta_div"] = eta_div
            last_results["eta_fric"] = eta_fric
            last_results["eta_total"] = eta_total

            lbl_result.config(
                text=f"score: eta_total={eta_total:.4f} (div={eta_div:.4f}, fric={eta_fric:.4f}, turn={eta_turn:.4f})",
                foreground="chartreuse4"
            )
            return eta_total

    except Exception as e:
        lbl_result.config(text=f"Error: {str(e)}", foreground="red")


# ============================================================
# GUI layout
# ============================================================
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

labels = [
    ("Combustion chamber temperature (K):", "3300"),
    ("Combustion chamber pressure (bar):", "30"),
    ("Mixture ratio (O/F):", "6.5"),
    ("Throat Radius (m):", "0.01531"),
    ("Expansion Ratio:", "5.6"),
    ("Reference Conical Nozzle Angle (deg):", "15"),
    ("Parabola Multiplier:", "1.5"),
    ("Initial Supersonic Arc Angle (deg):", "20"),
    ("Initial Straight Line Angle (deg):", "45"),
    ("Chamber Diameter (m):", "0.096"),
    ("Bell Contour (%):", "90"),
]

entries = []
for i, (text, default) in enumerate(labels):
    ttk.Label(frame, text=text).grid(row=i, column=0, sticky="w")
    e = ttk.Entry(frame)
    e.insert(0, default)
    e.grid(row=i, column=1)
    entries.append(e)

(entry_Tc, entry_Pc, entry_MR, entry_rt, entry_exp,
 entry_halfangle, entry_parabmultiplier, entry_theta_in, entry_theta_sub, entry_Rchamber,
 entry_bell_contour) = entries

def plot_sensitivity_theta_in():
    k_list = [0.05, 0.10, 0.20]
    theta_min = 15.0
    theta_max = 45.0
    N = 25
    theta_grid = np.linspace(theta_min, theta_max, N)

    theta_in_original = entry_theta_in.get()

    plt.figure(figsize=(10, 6))

    for k in k_list:
        etas = []

        last_results["k_turn_override"] = float(k)

        for th in theta_grid:
            entry_theta_in.delete(0, tk.END)
            entry_theta_in.insert(0, f"{th:.6f}")

            eta_total = run_simulation("score")
            etas.append(float(eta_total) if eta_total is not None else np.nan)

        plt.plot(theta_grid, np.array(etas, dtype=float), marker="o", label=f"k={k:.2f}")

    entry_theta_in.delete(0, tk.END)
    entry_theta_in.insert(0, theta_in_original)
    if "k_turn_override" in last_results:
        del last_results["k_turn_override"]

    plt.grid()
    plt.xlabel("theta_in (deg)")
    plt.ylabel("eta_total")
    plt.title("Sensitivity: eta_total vs theta_in for different k")
    plt.legend()
    plt.tight_layout()
    plt.show()

ttk.Button(frame, text="Plot 2D", command=lambda: run_simulation("2d")).grid(row=30, column=0)
ttk.Button(frame, text="Mach Curve", command=lambda: run_simulation("Mach")).grid(row=30, column=1)
ttk.Button(frame, text="Temperature Profile", command=lambda: run_simulation("Temp")).grid(row=31, column=0)
ttk.Button(frame, text="Pressure Profile", command=lambda: run_simulation("Pres")).grid(row=31, column=1)
ttk.Button(frame, text="Update CSV", command=lambda: run_simulation("csv")).grid(row=32, column=0)
ttk.Button(frame, text="Thermal (hg & q'')", command=lambda: run_simulation("thermal")).grid(row=32, column=1)
ttk.Button(frame, text="Friction (BL+CEA mu)", command=lambda: run_simulation("friction_bl_cea")).grid(row=33, column=0)
ttk.Button(frame, text="Calibrate k (turn)", command=lambda: run_simulation("calibrate_k")).grid(row=33, column=1)
ttk.Button(frame, text="Sensitivity: theta_in vs k", command=plot_sensitivity_theta_in).grid(row=34, column=0, columnspan=2, pady=6)

lbl_result = ttk.Label(frame, text="")
lbl_result.grid(row=35, column=0, columnspan=2, pady=4)

warning_label = tk.Label(root, text="", fg="red")
warning_label.grid(row=1, column=0, pady=10)

root.mainloop()