import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from rocketcea.cea_obj import CEA_Obj
from rocketcea.cea_obj import CEA_Obj, add_new_fuel

# Adding new fuel
card_str = """
fuel C30H62  C 30 H 62  wt%=83.00
h,cal=-158348.0  t(k)=298.15  rho=0.775
"""

add_new_fuel('Paraffin', card_str)
C = CEA_Obj(propName='', oxName='N2O', fuelName='Paraffin')

# Create main window
root = tk.Tk()
root.title("Nozzle Profile Generator")

# Main simulation function
def run_simulation(mode):
    try:
        # User Inputs
        Tc = float(entry_Tc.get())
        Pc = float(entry_Pc.get())
        MR = float(entry_MR.get())
        rt = float(entry_rt.get())
        exp = float(entry_exp.get())
        halfangle = math.radians(float(entry_halfangle.get()))
        theta_in = math.radians(float(entry_theta_in.get()))
        theta_sub = math.radians(float(entry_theta_sub.get()))
        R_chamber = float(entry_Rchamber.get()) / 2
        bell_contour = float(entry_bell_contour.get()) / 100
        cr = (R_chamber**2) / (rt**2)

        # Arc intersection values
        xr = -1.5 * rt * math.sin(theta_sub)
        yr = 2.5 * rt - 1.5 * rt * math.cos(theta_sub)

        if R_chamber < yr:
            warning_label.config(text="Chamber Diameter too Small for Selected Angle!", fg="red")
            return

        m = -math.tan(theta_sub)
        b = yr - m * xr
        xf = (R_chamber - b) / m

        def line(x): return m * x + b

        # Supersonic arc and parabola
        re = math.sqrt(exp) * rt
        Lcone = (re - rt) / math.tan(halfangle)
        Lparab = bell_contour * Lcone

        r_sup = 0.4 * rt
        yc_sup = 1.4 * rt
        angle_vals = np.linspace(math.pi / 2, math.pi / 2 - theta_in, 100)
        arc_x = r_sup * np.cos(angle_vals)
        arc_y = yc_sup - r_sup * np.sin(angle_vals)

        Px = arc_x[-1]
        Py = arc_y[-1]
        m0 = math.tan(theta_in)
        Lfinal = Px + Lparab

        # Calculating Coefficients
        A = np.array([
            [Px**2, Px, 1],  # y(Px) = Py
            [2*Px, 1, 0],    # y'(Px) = tan(theta_in)
            [Lfinal**2, Lfinal, 1]  # y(Lfinal) = re
        ])
        Y = np.array([Py, m0, re])
        coeffs = np.linalg.solve(A, Y)
        a, b_parab, c = coeffs

        # Inverted Concavity Exception
        if a > 0:
            warning_label.config(text="Inverted Parabola (a > 0)", fg="red")
            return
        else:
            warning_label.config(text="")

        # Defining Parabolic Equation
        x_vals = np.linspace(Px, Lfinal, 300)
        y_vals = a * x_vals**2 + b_parab * x_vals + c

        # Exceeding Exit Radius Exception
        epsilon = 1e-4
        if np.any(y_vals > re + epsilon):
            warning_label.config(text="Parabola Exceeds Exit Radius! \n" \
            f"Max y Value: {np.max(y_vals)} \n"
            f"Exit Radius: {re}", fg="red")
            return
        
        # Calculating the mach curve
        mw, gamma = C.get_Chamber_MolWt_gamma(Pc=Pc, MR=MR)

        def find_y_from_x(x):
            # SUPPRESSED: x coordinate is already translated. 
            # You MUST use the same geometry conditions used earlier in the nozzle profile.

            # supersonic parabola region
            if x >= (Px + translation_x):
                y = a * (x - translation_x)**2 + b_parab * (x - translation_x) + c

            # supersonic arc
            elif x < (Px + translation_x) and x >= translation_x:
                # compute y on the arc
                # parameter t corresponds to angle_vals indexing
                y = 1.4*rt - math.sqrt((0.4*rt)**2 - (x-translation_x)**2)

            # subsonic arc
            elif x < translation_x and x >= (xr + translation_x):
                y = 2.5*rt - math.sqrt(2.25*rt**2 - (x-translation_x)**2)

            # subsonic line
            else:
                y = m*(x-translation_x) + b

            return float(y)

        
        def find_mach(x):
            y = find_y_from_x(x)

            target = (y / rt)**2

            def func(Ma):
                return (1/Ma) * (2/(gamma+1)*(1+(gamma-1)/2*Ma**2))**((gamma+1)/(2*(gamma-1))) - target
            if x >= translation_x:
                Ma_guess = 2.0
            if x < translation_x:
                Ma_guess = 0.2
            Mach = fsolve(func, Ma_guess)[0]
            return float(Mach)
        
        def find_temp(Ma):

            target = Tc/(1+(gamma-1)/2 * Ma**2)

            def func(T):
                return T - target
            T_guess = 2000
            Ta = fsolve(func, T_guess)[0]
            return float(Ta)
        
        def find_pres(Ma):

            target = Pc * (1/(1+(gamma-1)/2 * Ma**2))**(gamma/(gamma-1))

            def func(P):
                return P - target
            P_guess = 30
            Pa = fsolve(func, P_guess)[0]
            return float(Pa)

        # Smoothed subsonic arc
        def arc_sub(x):
            val = 2.25 * rt**2 - x**2
            return 2.5 * rt - np.sqrt(val) if val >= 0 else None

        xs1, ys1 = [], []
        x = 0
        while x > xr:
            val = arc_sub(x)
            if val is not None:
                xs1.append(x)
                ys1.append(val)
            x -= 0.0001

        xs0 = np.linspace(xf, xr, 100)
        ys0 = line(xs0)
        
        # Apply horizontal translation to move the nozzle to the right
        translation_x = -xr + (R_chamber - yr) / math.tan(theta_sub)

        # Translation Application Function
        def correct_translation(x_vals, translation_x):
            # Applicating the Translation to All Values
            x_vals = [x + translation_x for x in x_vals]
            
            # Verifying if the First Value is Close to 0,
            # If so, it Forces the First Value to be 0
            if abs(x_vals[0]) < 1e-6:
                x_vals[0] = 0  
            
            return x_vals

        # Calcular a translação
        translation_x = -xr + (R_chamber - yr) / math.tan(theta_sub)

        # Aplicar a correção de translação nos valores de xs0, xs1, arc_x e x_vals
        xs0 = correct_translation(xs0, translation_x)
        xs1 = correct_translation(xs1, translation_x)
        arc_x = correct_translation(arc_x, translation_x)
        x_vals = correct_translation(x_vals, translation_x)

        # 2D Plot
        def plot2d():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, ys0, 'r--', label="Initial Straight Line")
            plt.plot(xs1, ys1, 'b', label="Subsonic Arc")
            plt.plot(arc_x, arc_y, 'g', label='Supersonic Arc')
            plt.plot(x_vals, y_vals, 'k', label='Parabolic Contour')
            plt.legend()
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            length = Lfinal - xf
            # Derivative of the Parabolic Equation: y'(x) = 2*a*x + b
            theta_out = math.atan(2*a*re + b)
            theta_out = -math.degrees(theta_out)  # Rad to Deg
            plt.title('2D Nozzle Profile', pad=30)
            plt.figtext(0.1, 0.94, f'Contraction Ratio  = {cr:.3f}', fontsize=10, ha='left')
            plt.figtext(0.1, 0.91, f'Expansion Ratio = {exp:.3f}', fontsize=10, ha='left')
            plt.figtext(0.7, 0.94, f'Nozzle Length = {length:.3f} m', fontsize=10, ha='left')
            plt.figtext(0.7, 0.91, f'Parabola Exit Angle = {theta_out:.3f}$^\\circ$', fontsize=10, ha='left')
            if re > R_chamber:
                plt.ylim(0, re*1.2)
            else:
                plt.ylim(0, R_chamber*1.2)
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show()

        def plot3d_single():
            x_vals_3d = np.concatenate([xs1, xs0, arc_x, x_vals])
            y_vals_3d = np.concatenate([ys1, ys0, arc_y, y_vals])
            mask_valid = np.isfinite(x_vals_3d) & np.isfinite(y_vals_3d)
            x_vals_3d = x_vals_3d[mask_valid]
            y_vals_3d = y_vals_3d[mask_valid]
            sort_idx = np.argsort(x_vals_3d)
            x_vals_3d = x_vals_3d[sort_idx]
            y_vals_3d = y_vals_3d[sort_idx]

            theta = np.linspace(0, 2 * np.pi, 200)
            X_mesh, Theta_mesh = np.meshgrid(x_vals_3d, theta, indexing="ij")
            R_mesh = np.tile(y_vals_3d, (theta.size, 1)).T
            Y_mesh = R_mesh * np.cos(Theta_mesh)
            Z_mesh = R_mesh * np.sin(Theta_mesh)

            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='plasma', edgecolor='none')
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Radius Y (m)')
            ax.set_zlabel('Radius Z (m)')
            ax.set_title('Single 3D Nozzle Profile', pad=0)
            ax.view_init(elev=10, azim=45)
            mng = plt.get_current_fig_manager()
            try:
                mng.window.state('zoomed')
            except:
                try:
                    mng.full_screen_toggle()
                except:
                    pass
            plt.show()

        # Multi-view 3D Plot
        def plot3d_multi():
            x_vals_3d = np.concatenate([xs1, xs0, arc_x, x_vals])
            y_vals_3d = np.concatenate([ys1, ys0, arc_y, y_vals])
            mask_valid = np.isfinite(x_vals_3d) & np.isfinite(y_vals_3d)
            x_vals_3d = x_vals_3d[mask_valid]
            y_vals_3d = y_vals_3d[mask_valid]
            sort_idx = np.argsort(x_vals_3d)
            x_vals_3d = x_vals_3d[sort_idx]
            y_vals_3d = y_vals_3d[sort_idx]

            theta = np.linspace(0, 2 * np.pi, 180)
            X_mesh, Theta_mesh = np.meshgrid(x_vals_3d, theta, indexing="ij")
            R_mesh = np.tile(y_vals_3d, (theta.size, 1)).T
            Y_mesh = R_mesh * np.cos(Theta_mesh)
            Z_mesh = R_mesh * np.sin(Theta_mesh)

            fig = plt.figure(figsize=(14, 10))
            views = [(30, 45, 'Isometric'), (90, 0, 'Side'), (0, 0, 'Front'), (0, 90, 'Top')]
            for i, (elev, azim, title) in enumerate(views, 1):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='plasma', edgecolor='none')
                ax.set_title(f"{title} View")
                ax.view_init(elev=elev, azim=azim)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.set_box_aspect([1, 1, 1])
            fig.suptitle('3D Nozzle Views', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            mng = plt.get_current_fig_manager()
            try:
                mng.window.state('zoomed')
            except:
                try:
                    mng.full_screen_toggle()
                except:
                    pass
            plt.show()

        

        def update_csv():
            # Combining Each Points List
            x_combined = [xs1, xs0, arc_x, x_vals]
            y_combined = [ys1, ys0, arc_y, y_vals]
            Ma_combined = [Mas1, Mas0, Ma_arc, Ma_vals]
            Ta_combined = [Tas1, Tas0, Ta_arc, Ta_vals]
            Pa_combined = [Pas1, Pas0, Pa_arc, Pa_vals]
            total_points = 100  # ou outro valor que preferires

            # Distributing the Points Equally
            lengths = [len(arr) for arr in x_combined]
            total_len = sum(lengths)

            x_raw = []
            y_raw = []
            Ma_raw = []
            Ta_raw = []
            Pa_raw = []

            for x_seg, y_seg, Ma_seg, Ta_seg, Pa_seg, seg_len in zip(x_combined, y_combined, Ma_combined, Ta_combined, Pa_combined, lengths):
                # Creating Step and Deffining a Minimum Step (=2)
                n_points = max(2, int((seg_len / total_len) * total_points))
                # Creating index (int) Equally Spaced
                indices = np.linspace(0, seg_len - 1, n_points).astype(int)

                x_raw.extend([x_seg[i] for i in indices])
                y_raw.extend([y_seg[i] for i in indices])
                Ma_raw.extend([Ma_seg[i] for i in indices])
                Ta_raw.extend([Ta_seg[i] for i in indices])
                Pa_raw.extend([Pa_seg[i] for i in indices])

            # Minimum Distance in mm
            min_dist_mm = 1
    	    
            # Checking if There's Points to Add
            if len(x_raw) > 0:
                # If There's Points, the First Points is Always Accepted and Added to the Filtered List
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

                # To Filter the Rest of the Points, a the Rest of the Points are Filtered (Expect the First Points, Already Accepted)
                for xi, yi, Mai, Tai, Pai in zip(x_raw[1:], y_raw[1:], Ma_raw[1:], Ta_raw[1:], Pa_raw[1:]):
                    # Getting the Distance Between the Actual Point and the Last Point
                    dist = math.hypot((xi - last_x) * 1000, (yi - last_y) * 1000, (Mai - last_Ma) * 1000, (Tai - last_Ta) * 1000, (Pai - last_Pa) * 1000) 
                    if dist >= min_dist_mm:
                        # If the Distance is Bigger than the Minimum Distance the Point is Added and the Last Point is Updated
                        x_filtered.append(xi)
                        y_filtered.append(yi)
                        Ma_filtered.append(Mai)
                        Ta_filtered.append(Tai)
                        Pa_filtered.append(Pai)
                        last_x = xi
                        last_y = yi
                        last_Ma = Mai
                        last_Ta = Tai
                        last_Pa = Pai

                x_final = x_filtered
                y_final = y_filtered
                Ma_final = Ma_filtered
                Ta_final = Ta_filtered
                Pa_final = Pa_filtered

            # Ensuring the Last Point is Correct (Lfinal, re)
            if x_final[-1] != Lfinal or y_final[-1] != re or Ma_final[-1] != find_mach(Lfinal) or Ta_final[-1] != find_temp(Ma_final) or Pa_final[-1] != find_pres(Ma_final):
                x_final[-1] = round(Lfinal + translation_x, 4)
                y_final[-1] = round(re, 4)
                Ma_final[-1] = round(find_mach(x_final[-1]), 4)
                Ta_final[-1] = round(find_temp(Ma_final[-1]), 4)
                Pa_final[-1] = round(find_pres(Ma_final[-1]), 4)

            # Converting m to mm
            x_final_mm = [x * 1000 for x in x_final]
            y_final_mm = [y * 1000 for y in y_final]

            # Zipping and Sorting the Points
            combined = list(zip(x_final_mm, y_final_mm, Ma_final, Ta_final, Pa_final))
            combined.sort()  # sorts by x automatically
            x_final_sorted, y_final_sorted, Ma_final_sorted, Ta_final_sorted, Pa_final_sorted = zip(*combined)

            try:
                with open("nozzle_geometry.csv", "w") as file:
                    file.write("x,y,z\n")
                    for x, y in zip(x_final_sorted, y_final_sorted):
                        file.write(f"{round(x,2)},{round(y,2)},0\n")
                lbl_result.config(text="CSV Updated Successfully!", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Erro: {str(e)}", foreground="red")


            try:
                with open("mach_temp_pres_profile.csv", "w") as f:
                    f.write("x_mm,Ma,T,P\n")
                    for x, Ma, T, P in zip(x_final_sorted, Ma_final_sorted, Ta_final_sorted, Pa_final_sorted):
                        f.write(f"{x:.2f},{Ma:.4f},{T:.6f},{P:.8f}\n")
                lbl_result.config(text="CSV Updated Successfully!", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Erro: {str(e)}", foreground="red")

        xs0 = np.array(xs0)
        Mas0 = np.array([find_mach(x) for x in xs0])
        Tas0 = np.array([find_temp(Ma) for Ma in Mas0])
        Pas0 = np.array([find_pres(Ma) for Ma in Mas0])
        xs1 = np.array(xs1)
        Mas1 = np.array([find_mach(x) for x in xs1])
        Tas1 = np.array([find_temp(Ma) for Ma in Mas1])
        Pas1 = np.array([find_pres(Ma) for Ma in Mas1])
        arc_x = np.array(arc_x)
        Ma_arc = np.array([find_mach(x) for x in arc_x])
        Ta_arc = np.array([find_temp(Ma) for Ma in Ma_arc])
        Pa_arc = np.array([find_pres(Ma) for Ma in Ma_arc])
        x_vals = np.array(x_vals)
        Ma_vals = np.array([find_mach(x) for x in x_vals])
        Ta_vals = np.array([find_temp(Ma) for Ma in Ma_vals])
        Pa_vals = np.array([find_pres(Ma) for Ma in Ma_vals])


        def plot_Mach():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Mas0, 'r--', label="Initial Straight Line")
            plt.plot(xs1, Mas1, 'b', label="Subsonic Arc")
            plt.plot(arc_x, Ma_arc, 'g', label='Supersonic Arc')
            plt.plot(x_vals, Ma_vals, 'k', label='Parabolic Contour')
            plt.title('Mach Curve', pad=30)
            plt.figtext(0.8, 0.94, f'Starting Mach = {find_mach(0):.3f}', fontsize=10, ha='left')
            plt.figtext(0.8, 0.91, f'Exit Mach = {find_mach(x_vals[-1]):.3f}', fontsize=10, ha='left')
            plt.figtext(0.08, 0.94, f'Choking Point = {translation_x:.3f} m', fontsize=10, ha='left')
            plt.xlabel('x (m)')
            plt.ylabel('Ma')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        def plot_Temp():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Tas0, 'r--', label="Initial Straight Line")
            plt.plot(xs1, Tas1, 'b', label="Subsonic Arc")
            plt.plot(arc_x, Ta_arc, 'g', label='Supersonic Arc')
            plt.plot(x_vals, Ta_vals, 'k', label='Parabolic Contour')
            plt.title('Temperature Profile', pad=30)
            plt.figtext(0.68, 0.94, f'Main Chamber Temperature = {find_temp(0):.3f} K', fontsize=10, ha='left')
            plt.figtext(0.68, 0.91, f'Exit Temperature = {find_temp(Ma_vals[-1]):.3f} K', fontsize=10, ha='left')
            plt.figtext(0.08, 0.94, f'Temperature At Choking Point = {find_temp(find_mach(translation_x)):.3f} K', fontsize=10, ha='left')
            plt.xlabel('x (m)')
            plt.ylabel('T (K)')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        def plot_Pres():
            plt.figure(figsize=(10, 6))
            plt.plot(xs0, Pas0, 'r--', label="Initial Straight Line")
            plt.plot(xs1, Pas1, 'b', label="Subsonic Arc")
            plt.plot(arc_x, Pa_arc, 'g', label='Supersonic Arc')
            plt.plot(x_vals, Pa_vals, 'k', label='Parabolic Contour')
            plt.title('Pressure Profile', pad=30)
            plt.figtext(0.7, 0.94, f'Main Chamber Pressure = {find_pres(0):.3f} bar', fontsize=10, ha='left')
            plt.figtext(0.7, 0.91, f'Exit Pressure = {find_pres(Ma_vals[-1]):.3f} bar', fontsize=10, ha='left')
            plt.figtext(0.08, 0.94, f'Pressure At Choking Point = {find_pres(find_mach(translation_x)):.3f} bar', fontsize=10, ha='left')
            plt.xlabel('x (m)')
            plt.ylabel('P (bar)')
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


        if mode == '2d':
            plot2d()
        elif mode == '3dm':
            plot3d_multi()
        elif mode == '3ds':
            plot3d_single()
        elif mode == 'csv':
            update_csv()
        elif mode == 'Mach':
            plot_Mach()
        elif mode == 'Temp':
            plot_Temp()
        elif mode == 'Pres':
            plot_Pres()

    except Exception as e:
        lbl_result.config(text=f"Error: {str(e)}")


# GUI Layout
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

labels = [
    ("Combustion chamber temperature (K):", "3000"),
    ("Combustion chamber pressure (bar):", "50"),
    ("Mixture ratio (O/F):", "6.5"),
    ("Throat Radius (m):", "0.01548"),
    ("Expansion Ratio:", "5"),
    ("Reference Conical Nozzle Angle (deg):", "15"),
    ("Initial Supersonic Arc Angle (deg):", "30"),
    ("Initial Straight Line Angle (deg):", "60"),
    ("Chamber Diameter (m):", "0.12"),
    ("Bell Contour (%):", "80")
]

entries = []
for i, (text, default) in enumerate(labels):
    ttk.Label(frame, text=text).grid(row=i, column=0)
    e = ttk.Entry(frame)
    e.insert(0, default)
    e.grid(row=i, column=1)
    entries.append(e)

entry_Tc, entry_Pc, entry_MR, entry_rt, entry_exp, entry_halfangle, entry_theta_in, entry_theta_sub, entry_Rchamber, entry_bell_contour = entries


    
ttk.Button(frame, text="Plot 2D", command=lambda: run_simulation('2d')).grid(row=30, column=0)
ttk.Button(frame, text="Single 3D Plot", command=lambda: run_simulation('3ds')).grid(row=30, column=1)
ttk.Button(frame, text="4 Views 3D Plot", command=lambda: run_simulation('3dm')).grid(row=31, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Update CSV", command=lambda: run_simulation('csv')).grid(row=32, columnspan=2)
ttk.Button(frame, text="Mach Curve", command=lambda: run_simulation('Mach')).grid(row=34, column=0, sticky="w", padx=45)
ttk.Button(frame, text="Temperature Profile", command=lambda: run_simulation('Temp')).grid(row=34 , column=1)
ttk.Button(frame, text="Pressure Profile", command=lambda: run_simulation('Pres')).grid(row=34 , columnspan=2)

lbl_result = ttk.Label(frame, text="")
lbl_result.grid(row=33, columnspan=2)

lbl_result = ttk.Label(frame, text="", font=("Arial", 10))
lbl_result.grid(row=33, columnspan=2)  

warning_label = tk.Label(root, text="", fg="red")
warning_label.grid(row=1, column=0, pady=10)

root.mainloop()

