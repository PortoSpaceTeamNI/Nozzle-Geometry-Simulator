import tkinter as tk
from tkinter import messagebox
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
import numpy as np
import math

# Personalized Fuel
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
fuel C8H8  C 8 H 8  wt%=7.00
h,cal=36209  t(k)=298.15  rho=0.906
fuel C4H6  C 4 H 6  wt%=5.00
h,cal=26744  t(k)=298.15  rho=0.6149
fuel C3H3N  C 3 H 3 N 1  wt%=5.00
h,cal=35157  t(k)=298.15  rho=0.810
"""

add_new_fuel('Paraffin/ABS', card_str)
C = CEA_Obj(propName='', oxName='N2O', fuelName='Paraffin/ABS')

# Calculation Functions with CEA
def get_gamma(Pc, OF, suparea):
    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    for line in s.split("\n"):
        if "GAMMA" in line:
            values = [float(val) for val in line.split() if val.replace('.', '', 1).isdigit()]
            return values[1]

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
