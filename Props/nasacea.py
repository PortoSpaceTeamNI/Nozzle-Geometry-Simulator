# exp.py
# -----------------------------------------------------------------------------
# RocketCEA helpers for:
#  - custom Paraffin fuel card
#  - p_exit(Pc, OF, eps) in atm  (as you already use)
#  - Isp(Pc, OF, eps) in s
#  - Cstar(Pc, OF, eps) in m/s  (from 'siunits' output)
#  - ISA + weighted ambient pressure (report method)
#
# Notes:
#  - Pc is in bar
#  - Pamb is in atm for the expansion solver workflow (matches your exp code)
# -----------------------------------------------------------------------------

import csv
from datetime import datetime

import matplotlib.pyplot as plt
from rocketcea.cea_obj import CEA_Obj, add_new_fuel


# --- Personalized Fuel ---
card_str = """
fuel C30H62  C 30 H 62  wt%=87.73523
h,cal=-191921.61  t(k)=298.15

fuel C8H8    C 8 H 8    wt%=4.64109
h,cal=35444.55  t(k)=298.15

fuel C4H6    C 4 H 6    wt%=3.31184
h,cal=26290.63  t(k)=298.15

fuel C3H3N   C 3 H 3 N 1  wt%=3.31184
h,cal=35156.79  t(k)=298.15

fuel C       C 1        wt%=1.00000
h,cal=0.0    t(k)=298.15

"""

add_new_fuel("Paraffin", card_str)

# CEA object
C = CEA_Obj(propName="", oxName="N2O", fuelName="Paraffin")


# -----------------------------------------------------------------------------
# Internal parsing utility (robust to formatting)
# -----------------------------------------------------------------------------

def _floats_in_line(line: str):
    vals = []
    for tok in line.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except Exception:
            pass
    return vals


# -----------------------------------------------------------------------------
# Primary quantities extracted from full CEA output (SI units requested)
# -----------------------------------------------------------------------------

def p_exit(Pc_bar: float, OF: float, eps: float) -> float:
    """
    Exit pressure in atm (as used in your plots / CSV).
    Pc in bar. Returns atm.

    Keeps your original logic (3rd numeric entry on the 'P,' line),
    but adds checks and a clear failure mode.
    """
    full = C.get_full_cea_output(
        Pc=Pc_bar, MR=OF, eps=eps,
        subar=None, short_output=0,
        pc_units="bar", output="siunits"
    )
    p_exits = []
    for line in full.split("\n"):
        if "P," in line:
            p_exits.extend(_floats_in_line(line))

    # Your original: return p_exits[2]
    if len(p_exits) < 3:
        raise ValueError("Could not parse exit pressure from CEA output (P, line).")
    return float(p_exits[2])


def Isp(Pc, OF, supar):
    vals = []
    full_output = C.get_full_cea_output(
        Pc=Pc, MR=OF, eps=supar,
        subar=None, short_output=0,
        pc_units='bar', output='siunits'
    )
    for line in full_output.split('\n'):
        if 'Isp,' in line:
            vals = []
            for v in line.split():
                try:
                    vals.append(float(v))
                except:
                    pass
            break
    # O valor que estás a apanhar está em m/s
    Isp_m_per_s = vals[1]
    Isp_seconds = Isp_m_per_s / 9.80665

    return Isp_seconds



def Cstar(Pc_bar: float, OF: float, eps: float) -> float:
    """
    c* from the 'CSTAR,' line (SI units output). Returns first numeric value.
    """
    full = C.get_full_cea_output(
        Pc=Pc_bar, MR=OF, eps=eps,
        subar=None, short_output=0,
        pc_units="bar", output="siunits"
    )
    for line in full.split("\n"):
        if "CSTAR," in line:
            vals = _floats_in_line(line)
            if vals:
                return float(vals[1])
            break
    raise ValueError("Could not parse Cstar from CEA output (CSTAR, line).")

def get_T_comb(Pc, OF, suparea):
    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    for line in s.split("\n"):
        if "T, K" in line:
            values = [float(val) for val in line.split() if val.replace('.', '', 1).isdigit()]
            return values[0]

def get_Molar_Mass(Pc, OF, suparea):
    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    for line in s.split("\n"):
        if "M," in line:
            values = [float(val) for val in line.split() if val.replace('.', '', 1).isdigit()]
            return values[1]
        
def get_R(Pc, OF, suparea):
    return 8.314 / (get_Molar_Mass(Pc, OF, suparea)/1000)

def get_gamma(Pc_bar: float, OF: float, eps: float) -> float:
    """
    c* from the 'CSTAR,' line (SI units output). Returns first numeric value.
    """
    full = C.get_full_cea_output(
        Pc=Pc_bar, MR=OF, eps=eps,
        subar=None, short_output=0,
        pc_units="bar", output="siunits"
    )
    for line in full.split("\n"):
        if "GAMMAs" in line:
            vals = _floats_in_line(line)
            if vals:
                #print(vals)
                return float(vals[1])
            break
    raise ValueError("Could not parse Cganna from CEA output (CSTAR, line).")



# -----------------------------------------------------------------------------
# ISA (troposphere) + weighted ambient pressure (report method)
# -----------------------------------------------------------------------------

def isa_troposphere_pressure_bar(h_m: float) -> float:
    """
    ISA troposphere (h < 11 km):
      P(h)=P0*(1 - L*h/T0)^(g0/(R*L))

    Returns pressure in bar.
    """
    # constants from your report page
    P0_Pa = 101325.0
    T0_K = 288.15
    L = 0.0065          # K/m
    g0 = 9.80665
    R = 287.05          # J/kg/K (air)

    if h_m < 0:
        h_m = 0.0
    if h_m > 11000.0:
        h_m = 11000.0  # clamp to model validity for this simplified sizing

    term = 1.0 - (L * h_m) / T0_K
    P_Pa = P0_Pa * (term ** (g0 / (R * L)))
    return P_Pa / 1e5   # bar


def burnout_altitude_m(tburn_s: float, thrust_N: float, m0_kg: float = 45.0) -> float:
    """
    Report-style quick estimate with v(0)=0 and constant accel:
      h_b = 0.5 * a * t_b^2
    with a ≈ (F/m0 - g0).

    m0_kg default 45 kg (consistent with the 56.86 m/s² example in your report).
    """
    g0 = 9.80665
    a = (thrust_N / m0_kg) - g0
    if a < 0:
        a = 0.0
    return 0.5 * a * (tburn_s ** 2)


def pamb_weighted_atm(tburn_s: float, thrust_N: float, m0_kg: float = 45.0) -> float:
    """
    Weighted design ambient pressure (report method):

      Pamb_design = 0.60*Pamb,0 + 0.40*Pamb,end

    where:
      Pamb,0 = ISA at h=0
      Pamb,end = ISA at burnout altitude estimate

    Returned in atm (because your exp workflow is in atm).
    """
    P0_bar = 1.01325
    hb_m = burnout_altitude_m(tburn_s, thrust_N, m0_kg=m0_kg)
    Pend_bar = isa_troposphere_pressure_bar(hb_m)

    Pdesign_bar = 0.6 * P0_bar + 0.4 * Pend_bar
    #print(f"Pamb={Pdesign_bar} bar")
    # bar -> atm
    return Pdesign_bar / 1.01325


# -----------------------------------------------------------------------------
# Optional: solve eps for Pexit=Pamb (no GUI, used by MC driver)
# -----------------------------------------------------------------------------

def solve_eps_for_pamb(Pc_bar: float, OF: float, Pamb_atm: float,
                       eps_min=0.5, eps_max=25.0, deps=0.5) -> float:
    """
    Find eps such that p_exit(Pc,OF,eps) crosses Pamb_atm.
    Linear interpolation between the two bracketing points.
    """
    prev_eps = None
    prev_pex = None

    eps = eps_min
    while eps <= eps_max + 1e-12:
        pex = p_exit(Pc_bar, OF, eps)  # atm

        if prev_eps is not None:
            crossed = ((prev_pex > Pamb_atm and pex < Pamb_atm) or
                       (prev_pex < Pamb_atm and pex > Pamb_atm))
            if crossed and (pex != prev_pex):
                eps_star = prev_eps + (Pamb_atm - prev_pex) * (eps - prev_eps) / (pex - prev_pex)
                #print(eps_star)
                return float(eps_star)

        prev_eps, prev_pex = eps, pex
        eps += deps

    return float("nan")