import tkinter as tk
from tkinter import messagebox
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
import numpy as np
import math
import re

# Personalized Fuel
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


add_new_fuel('Paraffin', card_str)
C = CEA_Obj(propName='', oxName='N2O', fuelName='Paraffin')

# Calculation Functions with CEA
def get_gamma(Pc, OF, suparea):
    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    #print(s)
    for line in s.split("\n"):
        if "GAMMA" in line:
            values = [float(val) for val in line.split() if val.replace('.', '', 1).isdigit()]
            print(values)
            return values[1]

def get_gamma_triplet(Pc_bar, OF, eps):
    s = C.get_full_cea_output(Pc=Pc_bar, MR=OF, eps=eps,
                              output='siunits', pc_units='bar')
    for line in s.splitlines():
        u = line.strip().upper()
        if u.startswith("GAMMA"):
            vals = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)]
            if len(vals) >= 3:
                return vals[0], vals[1], vals[2]  # chamber, throat, exit
    raise RuntimeError("Não encontrei GAMMA no output CEA.")

_FLOAT_RE = re.compile(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?')

def _nums(line: str):
    return [float(x) for x in _FLOAT_RE.findall(line)]

def get_gas_transport_props(Pc, OF, suparea,
                            col=1,  # 0=chamber, 1=throat, 2=exit (mesma lógica do teu get_gamma)
                            default_mu=3.5e-5, default_Cp=3000.0, default_Pr=0.7, default_cstar=1500.0,
                            debug=False):
    """
    Input:
      Pc       [bar]
      OF       [-]  (MR no RocketCEA)
      suparea  [-]  (eps = Ae/At)

    Output:
      mu    [Pa.s]
      Cp    [J/kg/K]
      Pr    [-]
      cstar [m/s]
    """
    mu = default_mu
    Cp = default_Cp
    Pr = default_Pr
    cstar = default_cstar

    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    if debug:
        print(s)

    for line in s.splitlines():
        u = line.upper()

        # ---- c* ----
        if ("CSTAR" in u) and ("M/SEC" in u or "M/S" in u or "M-SEC" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                cstar = vals[col] if col < len(vals) else vals[-1]

        # ---- viscosity ----
        # aparece tipicamente como "VISC" / "VISCOSITY" com 3 colunas (c, t, e)
        if ("VISC" in u) or ("VISCOSITY" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                mu_val = vals[col] if col < len(vals) else vals[-1]

                # conversões defensivas caso a tua build não esteja mesmo em Pa·s
                if "MICROPOISE" in u:     # 1 microPoise = 1e-7 Pa·s
                    mu_val *= 1e-7
                elif "MILLIPOISE" in u:   # 1 mPoise = 1e-4 Pa·s
                    mu_val *= 1e-4
                elif "POISE" in u and "PA" not in u:  # 1 Poise = 0.1 Pa·s
                    mu_val *= 0.1

                mu = mu_val

        # ---- Cp ----
        # linhas típicas: "Cp, KJ/(KG)(K)   ...  ...  ..."
        if (u.strip().startswith("CP") or " CP" in u) and ("KG" in u) and ("K" in u):
            vals = _nums(line)
            if len(vals) >= 3:
                cp_val = vals[col] if col < len(vals) else vals[-1]
                if "KJ" in u:
                    cp_val *= 1000.0  # kJ/kg/K -> J/kg/K
                Cp = cp_val

        # ---- Prandtl ----
        if "PRANDTL" in u:
            vals = _nums(line)
            if len(vals) >= 3:
                Pr = vals[col] if col < len(vals) else vals[-1]

    return mu, Cp, Pr, cstar

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

# Interface
def calculate_radius():
    try:
        mass_flow = float(entry_mass_flow.get())
        Pc = float(entry_pc.get())
        OF = float(entry_of.get())
        expansion_ratio = float(entry_expansion_ratio.get())

        R = 8.314 / (get_Molar_Mass(Pc, OF, expansion_ratio)/1000)  
        gamma = get_gamma(Pc, OF, expansion_ratio)
        Tcomb = get_T_comb(Pc, OF, expansion_ratio)

        area_throat = mass_flow / (Pc * 1e5 * np.sqrt(
            gamma * ((2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))) * (1 / (R * Tcomb))
        ))

        throat_radius = np.sqrt(area_throat / math.pi)
        messagebox.showinfo("Result", f"Throat Radius: {throat_radius:.5f} m")
    except Exception as e:
        messagebox.showerror("Error", f"Calculation failed:\n{e}")

# Generating Window
root = tk.Tk()
root.title("Rocket Nozzle Throat Radius Calculator")

# Inputs
tk.Label(root, text="Mass Flow (kg/s):").grid(row=0, column=0, sticky="e")
entry_mass_flow = tk.Entry(root)
entry_mass_flow.grid(row=0, column=1)

tk.Label(root, text="Chamber Pressure Pc (bar):").grid(row=1, column=0, sticky="e")
entry_pc = tk.Entry(root)
entry_pc.grid(row=1, column=1)

tk.Label(root, text="O/F Ratio:").grid(row=2, column=0, sticky="e")
entry_of = tk.Entry(root)
entry_of.grid(row=2, column=1)

tk.Label(root, text="Expansion Ratio:").grid(row=3, column=0, sticky="e")
entry_expansion_ratio = tk.Entry(root)
entry_expansion_ratio.grid(row=3, column=1)

calc_button = tk.Button(root, text="Calculate Throat Radius", command=calculate_radius)
calc_button.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
