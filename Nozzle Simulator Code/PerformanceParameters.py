import tkinter as tk
from tkinter import messagebox
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Addin New Customized Fuel
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

# Returning Dicionaire Parameters
def get_performance_params(Pc, OF, suparea):
    s = C.get_full_cea_output(Pc=Pc, MR=OF, eps=suparea, output='siunits', pc_units='bar')
    result = {}
    for line in s.split("\n"):
        if "Isp," in line:
            result['Specific Impulse, s'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][1]) / 9.81
        if "Isp," in line:
            result['Exhaust Velocity, m/s'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][1])
        elif "Ivac," in line:
            result['Specific Impulse in Vacuum, s'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][1]) / 9.81
        elif "CF," in line:
            result['Thrust Coefficient'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][1])
        elif "CSTAR," in line:
            result['Characteristic Velocity, m/s'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][1])
        elif "MACH" in line:
            result['Mach Number'] = float([val for val in line.split() if val.replace('.', '', 1).isdigit()][2])
    return result

def plot_Isp_and_table(OF1, OF2, OF3, Pc, Exp):
    OFs = [OF1, OF2, OF3]

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    # Creating Table
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')

    params = get_performance_params(Pc, OF1, Exp)
    table_data = [[k, f"{v:.3f}"] for k, v in params.items()]
    table = ax_table.table(cellText=table_data, colLabels=["Parameter", "Value"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 1.7)
    ax_table.text(0.01, 0.75, f'Performance @ Pc={Pc} bar, O/F={OF1}, Exp={Exp}',
         transform=ax_table.transAxes, fontsize=10, weight='bold')

    # Ploting
    ax_plot = fig.add_subplot(gs[1])

    for OF in OFs:
        Isps = []
        Ars = []
        Ar = 1.0
        while Ar < 25:
            Isps.append(get_performance_params(Pc, OF, Ar)['Specific Impulse, s'])
            Ars.append(Ar)
            Ar += 0.5
        ax_plot.plot(Ars, Isps, label=f'OF = {OF}')

    ax_plot.set_title(C.desc)
    ax_plot.set_xlabel('Expansion Ratio ($\\epsilon$)')
    ax_plot.set_ylabel('Specific Impulse (s)')
    ax_plot.grid(True)
    ax_plot.set_ylim(150, 300)
    ax_plot.legend()
    ax_plot.set_xticks(range(2, 26, 2))

    # Full Screen
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')
    except:
        try:
            mng.full_screen_toggle()
        except:
            pass

    plt.tight_layout()
    plt.show()


# Starting and Creating Interface
def start_gui():
    global root
    root = tk.Tk()
    root.title("Nozzle Performance Parameters")
    root.geometry("400x350")

    # Labels e Inputs
    tk.Label(root, text="Main Oxidizer/Fuel Ratio (OF):").pack()
    entry_of1 = tk.Entry(root)
    entry_of1.pack()

    tk.Label(root, text="Second Oxidizer/Fuel Ratio (OF):").pack()
    entry_of2 = tk.Entry(root)
    entry_of2.pack()

    tk.Label(root, text="Third Oxidizer/Fuel Ratio (OF):").pack()
    entry_of3 = tk.Entry(root)
    entry_of3.pack()

    tk.Label(root, text="Chamber Pressure (bar):").pack()
    entry_pc = tk.Entry(root)
    entry_pc.pack()

    tk.Label(root, text="Expansion Ratio:").pack()
    entry_exp = tk.Entry(root)
    entry_exp.pack()

    # Calling the Exhaust Velocity Ploting Function
    def plot_th():
        try:
            OF1 = float(entry_of1.get())
            OF2 = float(entry_of2.get())
            OF3 = float(entry_of3.get())
            Exp = float(entry_exp.get())
            Pc = float(entry_pc.get())

            # Bloquing Possible Looping
            root.after(100, lambda: plot_Isp_and_table(OF1, OF2, OF3, Pc, Exp))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    tk.Button(root, text="Get Performance Parameters", command=plot_th).pack(pady=10)

    root.mainloop()

# Executing
if __name__ == '__main__':
    start_gui()