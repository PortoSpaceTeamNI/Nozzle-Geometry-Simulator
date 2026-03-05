import tkinter as tk
from tkinter import messagebox
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
import matplotlib.pyplot as plt

import csv
from datetime import datetime

# --- Personalized Fuel ---
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

# --- Exit Pressure Calculation ---
def p_exit(Pc, OF, supar):
    p_exits = []
    full_output = C.get_full_cea_output(Pc=Pc, MR=OF, eps=supar, subar=None, short_output=0, pc_units='bar', output='siunits')
    for line in full_output.split('\n'):
        if 'P,' in line:
            values = line.split()
            for value in values:
                try:
                    p_exits.append(float(value))
                except:
                    pass
    return p_exits[2]

# --- Plot Creation and Analysis ---
def expansion_ratio(P1, P2, P3, OF, ambient_pressure):
    pcs = [P1, P2, P3]
    exp_pamb = []
    all_exps = []

    # ---- NEW: storage for main pressure curve (Pc == P1) ----
    main_curve = []  # list of dicts: {"eps":..., "p_exit_atm":...}

    for pc in pcs:
        pexits = []
        exps = []
        exp = 0.5
        prev_exp = None
        prev_p_exit = None

        while exp <= 25:
            p_exit_val = p_exit(pc, OF, exp)   # <- ensure units are consistent with your ylabel
            pexits.append(p_exit_val)
            exps.append(exp)
            all_exps.append(exp)

            # ---- NEW: store points only for main chamber pressure ----
            if pc == P1:
                main_curve.append({"eps": exp, "p_exit_atm": p_exit_val})

            if prev_exp is not None:
                crossed = ((prev_p_exit > ambient_pressure and p_exit_val < ambient_pressure) or
                           (prev_p_exit < ambient_pressure and p_exit_val > ambient_pressure))
                if crossed:
                    exp_interpolated = prev_exp + (ambient_pressure - prev_p_exit) * (exp - prev_exp) / (p_exit_val - prev_p_exit)
                    exp_pamb.append((pc, exp_interpolated))
                    all_exps.append(exp_interpolated)

            prev_exp = exp
            prev_p_exit = p_exit_val
            exp += 0.5

        plt.plot(exps, pexits, label=f'Pc = {pc} bar')

    # ---- NEW: write CSV for main pressure curve ----
    # pick the interpolated eps for P1, if it exists
    eps_at_pamb_P1 = None
    for pc, eps_int in exp_pamb:
        if pc == P1:
            eps_at_pamb_P1 = eps_int
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"pexit_vs_exp_Pc{P1:g}bar_OF{OF:g}_Pamb{ambient_pressure:g}atm_{timestamp}.csv"

    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pc_bar", "OF", "Pamb_atm", "eps_at_Pamb_interp"])
        writer.writerow([P1, OF, ambient_pressure, eps_at_pamb_P1])
        writer.writerow([])

        writer.writerow(["eps", "p_exit_atm"])
        for row in main_curve:
            writer.writerow([row["eps"], row["p_exit_atm"]])

    print(f"[CSV] Saved main curve to: {csv_name}")

    # ---- plotting (unchanged) ----
    if exp_pamb:
        xs = [exp for _, exp in exp_pamb]
        ys = [ambient_pressure] * len(exp_pamb)
        plt.scatter(xs, ys, color='red', marker='o', s=20, label=f"Pexit = {ambient_pressure} atm")
        plt.figtext(0.1, 0.94, f'Ideal Expansion Ratio (Pc={P1} bar, O/F={OF}, P_amb={ambient_pressure}) = {xs[0]:.3f}', fontsize=10, ha='left')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Nozzle Exit Pressure vs. Expansion Ratio')
    plt.xlabel('Expansion Ratio ($\\epsilon$)')
    plt.ylabel('Exit Pressure (atm)')
    plt.xticks(range(2, 26, 2))
    plt.ylim(0, max(1.5, ambient_pressure + 0.5))

    if all_exps:
        x_min = max(0, min(all_exps) - 1)
        x_max = max(all_exps) + 1
        plt.xlim(x_min, x_max)

    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')
    except:
        try:
            mng.full_screen_toggle()
        except:
            pass

    plt.show()

    for pc, exp in exp_pamb:
        print(f'Chamber Pressure = {pc} bar \nExit Pressure = Ambient Pressure ({ambient_pressure} atm) at Expansion Ratio = {exp}\n----===(+)===----')


# --- GUI Function ---
def start_gui():
    global root
    root = tk.Tk()
    root.title("Nozzle Expansion Ratio Analysis")
    root.geometry("400x350")

    # Labels e Inputs
    tk.Label(root, text="Main Chamber Pressure (bar):").pack()
    entry_p1 = tk.Entry(root)
    entry_p1.pack()

    tk.Label(root, text="Second Chamber Pressure (bar):").pack()
    entry_p2 = tk.Entry(root)
    entry_p2.pack()

    tk.Label(root, text="Third Chamber Pressure (bar):").pack()
    entry_p3 = tk.Entry(root)
    entry_p3.pack()

    tk.Label(root, text="O/F Ratio:").pack()
    entry_of = tk.Entry(root)
    entry_of.insert(0, "6.5")
    entry_of.pack()

    tk.Label(root, text="Ambient Pressure (atm):").pack()
    entry_ambient = tk.Entry(root)
    entry_ambient.insert(0, "1.0")  
    entry_ambient.pack()

    def plot_analysis():
        try:
            P1 = float(entry_p1.get())
            P2 = float(entry_p2.get())
            P3 = float(entry_p3.get())
            OF = float(entry_of.get())
            ambient_pressure = float(entry_ambient.get())

            
            root.after(100, lambda: expansion_ratio(P1, P2, P3, OF, ambient_pressure))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    tk.Button(root, text="Plot Nozzle Expansion Ratio Analysis", command=plot_analysis).pack(pady=10)

    root.mainloop()

# --- Execution ---
if __name__ == '__main__':
    start_gui()
