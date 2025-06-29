from rocketcea.cea_obj import CEA_Obj
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
from pylab import *
import matplotlib.pyplot as plt
import math
import tkinter as tk
import tkinter as messagebox

#Adding New Fuel
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
"""

add_new_fuel('Paraffin', card_str)
C = CEA_Obj(propName='', oxName='N2O', fuelName='Paraffin')


def thrustcoefficient(Pc, OF, supar):
    tcfs = []
    full_output = C.get_full_cea_output( Pc=Pc, # number or list of chamber pressures
                                        MR=OF,   # number or list of mixture ratios
                                        eps=supar,   # number or list of supersonic area ratios
                                        subar=None,     # number or list of subsonic area ratios
                                        short_output=0,  # 0 or 1 to control output length
                                        pc_units='bar', # pc_units = 'psia', 'bar', 'atm', 'mmh'
                                        output='siunits',# output = 'calories' or 'siunits'
                                        fac_CR=None)
    
    if "50 ITERATIONS DID NOT SATISFY CONVERGENCE" in full_output:
        print(f"Error: O CEA não convergiu para Pc={Pc}, exp={supar}. Ignorando este ponto.")
        return None  # Retorna None para evitar erro


    for line in full_output.split('\n'):
        if 'CF' in line:
            values = line.split()
            for value in values:
                try:
                    tcfs.append(float(value))
                except:
                    pass

    if len(tcfs)>1:
        return tcfs[1]
    else:
        return None
    
def plot_cf(OF1, OF2, OF3):
    OFs = [OF1, OF2, OF3]

    for OF in OFs:
        cfs = []
        exps = []
        exp = 0.05
        

        while exp <= 25:
            cfs.append(thrustcoefficient(30, OF, exp))
            exps.append(exp)
            exp += 0.05
        plt.plot(exps, cfs, label='OF=%g' %OF)
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Nozzle Thrust Coefficient vs. Nozzle Expansion Ratio')
    plt.xlabel('Expansion Ratio ($\\epsilon$)')
    plt.ylabel('Thrust Coefficient')
    plt.xticks(range(2, 26, 2))
    
    plt.show()

def thrustcalculation(Pc_bar, OF, supar, rt_m, opt):
    cf = thrustcoefficient(Pc_bar, OF, supar)
    if cf is None:
        return None
    
    thrust = Pc_bar * 10**5 * thrustcoefficient(Pc_bar, OF, supar) * math.pi * rt_m**2
    if opt == 0:
            return thrust
    if opt == 1:
            print('---==(+)==---')
            return print(f'For: \n | Pc={Pc_bar}bar \n | OF={OF}  \n | Exp={supar}  \n | Rt={rt_m}m  \n Thrust: {thrust}N \n ---==(+)==---')
    
def plot_thrust(P1, P2, P3, OF, expansion_ratio, rt):
    Pcs = [P1, P2, P3]
    for Pc in Pcs:
        Ths = []
        exps = []
        exp = 0.05
        

        while exp <= 25:
            Ths.append(thrustcalculation(Pc, OF, exp, rt, 0))
            exps.append(exp)
            exp += 0.05
        plt.plot(exps, Ths, label='Pc=%g bar' %Pc)

    thrust_calculated = thrustcalculation(P1, OF, expansion_ratio, rt, 0)

    plt.scatter(expansion_ratio, thrust_calculated, color='red', marker='o', s=20, label=f"Thrust Calculated \n(Chamber Pressure={P1} bar)")
    plt.figtext(0.06, 0.94, f'Thrust (Pc={P1} bar, O/F={OF}, Exp={expansion_ratio}, Throat Radius={rt} m) = {thrust_calculated:.3f} N', fontsize=10, ha='left')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Nozzle Thrust Production vs. Nozzle Expansion Ratio')
    plt.xlabel('Expansion Ratio ($\\epsilon$)')
    plt.ylabel('Thrust (N)')
    plt.xticks(range(2, 26, 2))
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')
    except:
            try:
                mng.full_screen_toggle()
            except:
                pass
    plt.show()

# Starting and Creating Interface
def start_gui():
    global root
    root = tk.Tk()
    root.title("Nozzle Thrust Generation Analysis")
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
    entry_of.pack()

    tk.Label(root, text="Throat Radius (m):").pack()
    entry_rt = tk.Entry(root)
    entry_rt.pack()

    tk.Label(root, text="Expansion Ratio:").pack()
    entry_exp = tk.Entry(root)
    entry_exp.pack()

    # Calling the Exhaust Velocity Ploting Function
    def plot_th():
        try:
            P1 = float(entry_p1.get())
            P2 = float(entry_p2.get())
            P3 = float(entry_p3.get())
            Exp = float(entry_exp.get())
            Rt = float(entry_rt.get())
            OF = float(entry_of.get())

            # Bloquing Possible Looping
            root.after(100, lambda: plot_thrust(P1, P2, P3, OF, Exp, Rt))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    tk.Button(root, text="Plot Thrust", command=plot_th).pack(pady=10)

    root.mainloop()

# Executing
if __name__ == '__main__':
    start_gui()

