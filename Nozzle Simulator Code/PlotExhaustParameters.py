from rocketcea.cea_obj import CEA_Obj
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_propellant
from pylab import *
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter as messagebox

#Adding New Fuel
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
"""

add_new_fuel('Paraffin', card_str)
C = CEA_Obj(propName='', oxName='N2O', fuelName='Paraffin')

def Mach(Pc, OF, suparea):
    s = C.get_full_cea_output( Pc=Pc, MR=OF, eps=suparea, subar=None, short_output=0, pc_units='bar', output='siunits', fac_CR=None) 
    for line in s.split("\n"):
            if "MACH " in line:  
                values = line.split()  
                Mach_values = []
                for l in values:
                    try:
                        Mach_values.append(float(l))
                    except ValueError:
                        pass
                    
                return float(Mach_values[2])
            
def Sonic_Velocity(Pc, OF, suparea):
    s = C.get_full_cea_output( Pc=Pc, MR=OF, eps=suparea, subar=None, short_output=0, pc_units='bar', output='siunits', fac_CR=None) 
    for line in s.split("\n"): 
            if "SON VEL" in line: 
                values = line.split()  
                a_values = []
                for l in values:
                    try:
                        a_values.append(float(l))
                    except ValueError:
                        pass
                    
                return float(a_values[2])

def T_exhaust(Pc, OF, suparea):
    s = C.get_full_cea_output( Pc=Pc, MR=OF, eps=suparea, subar=None, short_output=0, pc_units='bar', output='siunits', fac_CR=None) 
    for line in s.split("\n"):  
            if "T, K" in line:  
                values = line.split()  
                temp_values = []
                for l in values:
                    try:
                        temp_values.append(float(l))
                    except ValueError:
                        pass
                    
                return temp_values[2]
            
def plot_combined(OF1, OF2, OF3, Pc, exp):
    OFs = [OF1, OF2, OF3]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Two side-by-side plots

    # Left Plot - T_exhaust
    for OF in OFs:
        Texs = []
        Ars = []
        Ar = 1.0
        while Ar < 25:
            Texs.append(T_exhaust(Pc, OF, Ar))
            Ars.append(Ar)
            Ar += 0.5
        axs[0].plot(Ars, Texs, label='OF=%g' % OF)

    exhaust_temperature = T_exhaust(Pc, OF1, exp)
    axs[0].scatter(exp, exhaust_temperature, color = 'red', marker = 'o', s=20, label=f'Exhaust Temperature (Exp={exp})')
    axs[0].legend(loc='best')
    axs[0].grid(True)
    axs[0].set_title('Exhaust Temperature')
    axs[0].set_xlabel('Expansion Ratio ($\\epsilon$)')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].set_ylim(1000, 3500)
    axs[0].set_xticks(range(2, 25, 2))

    # Right Plot - Exhaust Velocity
    for OF in OFs:
        Ars = []
        v_exs = []
        Ar = 1
        while Ar < 25:
            Ars.append(Ar)
            v_exs.append(Mach(Pc, OF, Ar) * Sonic_Velocity(Pc, OF, Ar))
            Ar += 1
        axs[1].plot(Ars, v_exs, label='OF=%g' % OF)

    exhaust_velocity = Mach(Pc, OF1, exp) * Sonic_Velocity(Pc, OF1, exp)
    axs[1].scatter(exp, exhaust_velocity, color='red', marker='o', s=20, label=f"Exhaust Velocity (Exp={exp})")
    plt.figtext(0.13, 0.85, f'Exhaust Temperature (Pc={Pc} bar, O/F={OF1}, Exp={exp}) = {exhaust_temperature:.3f} K', fontsize=10, ha='left')
    axs[1].legend(loc='best')
    axs[1].grid(True)
    axs[1].set_title('Exhaust Velocity')
    axs[1].set_xlabel('Expansion Ratio ($\\epsilon$)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_ylim(1000, 3000)
    axs[1].set_xticks(range(2, 25, 2))

    fig.suptitle(C.desc)
    plt.figtext(0.62, 0.85, f'Exhaust Velocity (Pc={Pc} bar, O/F={OF1}, Exp={exp}) = {exhaust_velocity:.3f} m/s', fontsize=10, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Fullscreen window
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
    root.title("Nozzle Expansion Ratio Analysis")
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
    def plot_exV():
        try:
            OF1 = float(entry_of1.get())
            OF2 = float(entry_of2.get())
            OF3 = float(entry_of3.get())
            Pc = float(entry_pc.get())
            Exp = float(entry_exp.get())

            # Bloquing Possible Looping
            root.after(100, lambda: plot_combined(OF1, OF2, OF3, Pc, Exp))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    tk.Button(root, text="Plot Exhaust Velocity and \n Exhaust Temperature", command=plot_exV).pack(pady=10)

    root.mainloop()

# Executing
if __name__ == '__main__':
    start_gui()
