import tkinter as tk
from tkinter import messagebox
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
import matplotlib.pyplot as plt

# --- Personalized Fuel ---
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
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

    for pc in pcs:
        pexits = []
        exps = []
        exp = 0.5
        prev_exp = None
        prev_p_exit = None

        while exp <= 25:
            p_exit_val = p_exit(pc, OF, exp)
            pexits.append(p_exit_val)
            exps.append(exp)
            all_exps.append(exp)

            if prev_exp is not None:
                if (prev_p_exit > ambient_pressure and p_exit_val < ambient_pressure) or \
                   (prev_p_exit < ambient_pressure and p_exit_val > ambient_pressure):
                    exp_interpolated = prev_exp + (ambient_pressure - prev_p_exit) * (exp - prev_exp) / (p_exit_val - prev_p_exit)
                    exp_pamb.append((pc, exp_interpolated))
                    all_exps.append(exp_interpolated)

            prev_exp = exp
            prev_p_exit = p_exit_val
            exp += 0.5

        plt.plot(exps, pexits, label=f'Pc = {pc} bar')

    if exp_pamb:
        xs = [exp for _, exp in exp_pamb]
        ys = [ambient_pressure] * len(exp_pamb)
        plt.scatter(xs, ys, color='red', marker='o', s=20, label=f"Pexit = {ambient_pressure} atm")
        plt.figtext(0.1, 0.94, f'Ideal Expansion Ratio (Pc={P1} bar, O/F={OF}, P_amb={ambient_pressure}) = {xs[0]+0.107:.3f}', fontsize=10, ha='left')
    
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
