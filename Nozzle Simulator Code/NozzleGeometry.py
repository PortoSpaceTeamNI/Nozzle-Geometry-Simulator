import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from rocketcea.cea_obj import CEA_Obj
from rocketcea.cea_obj import CEA_Obj, add_new_fuel
from nozzle_friction import friction_bl_cea
from k import *
import re as re_mod


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

last_results = {}

def plot_friction_results(res, translation_x, rt):
    x = res["x_m"]
    r = res["r_m"]
    Ma_iso = res["Ma_iso"]
    Ma_eff = res["Ma_eff"]
    delta = res["delta_star_m"]
    r_eff = res["r_eff_m"]
    mu = res["mu_Pa_s"]
    Rex = res["Re_x_plot"]

    # Mach comparison
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

    # delta*
    plt.figure(figsize=(10, 6))
    plt.plot(x, delta * 1e3)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("δ* (mm)")
    plt.title("Boundary-layer displacement thickness (continuous)")
    plt.tight_layout()
    plt.show()

    # r vs r_eff
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

    # effective area loss
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

    # mu(x)
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

    # Re_x
    plt.figure(figsize=(10, 6))
    plt.plot(x, Rex)
    plt.axvline(translation_x, linestyle="--")
    plt.grid()
    plt.xlabel("x (m)")
    plt.ylabel("Re_x")
    plt.title("Re_x along nozzle (continuous x)")
    plt.tight_layout()
    plt.show()

    # print CEA triplet if available
    trip = res.get("cea_mu_triplet")
    if trip is not None:
        print("CEA mu triplet (T[K], mu[Pa.s]):",
              (trip["Tc"], trip["muc"]),
              (trip["Tt"], trip["mut"]),
              (trip["Te"], trip["mue"]))

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

        theta_out_rad = math.atan(2*a*Lfinal + b_parab)   # y'(x)=2ax+b, avaliar em x=Lfinal
        theta_out_deg = abs(math.degrees(theta_out_rad))  # magnitude (positivo)

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

                # ------------------------------------------------------------
        # Build full nozzle stations (x_all, r_all) for friction model
        # ------------------------------------------------------------
        x_all = np.concatenate([xs0, xs1, arc_x, x_vals]).astype(float)
        r_all = np.concatenate([ys0, ys1, arc_y, y_vals]).astype(float)

        mask = np.isfinite(x_all) & np.isfinite(r_all)
        x_all = x_all[mask]
        r_all = r_all[mask]
        idx = np.argsort(x_all)
        x_all = x_all[idx]
        r_all = r_all[idx]

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
            plt.title('2D Nozzle Profile', pad=30)
            plt.figtext(0.1, 0.94, f'Contraction Ratio  = {cr:.3f}', fontsize=10, ha='left')
            plt.figtext(0.1, 0.91, f'Expansion Ratio = {exp:.3f}', fontsize=10, ha='left')
            plt.figtext(0.7, 0.94, f'Nozzle Length = {length:.3f} m', fontsize=10, ha='left')
            plt.figtext(0.7, 0.91, f'Parabola Exit Angle = {theta_out_deg:.3f}$^\\circ$', fontsize=10, ha='left')
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

                # --- coloca isto perto do topo (global) ---
        last_results = {}  # guarda outputs da última simulação (para poderes usar noutro código)


        # --- dentro do run_simulation(mode), depois de resolveres a parábola (a, b_parab, c) ---
        # ATENÇÃO: no teu código tens bug aqui:
        #   theta_out = math.atan(2*a*re + b)
        # "b" aí é a variável da reta sub-sónica. O correto para derivada da parábola é b_parab.
        def compute_theta_out_deg(a, b_parab, x_exit):
            # y'(x) = 2 a x + b_parab
            return -math.degrees(math.atan(2.0 * a * x_exit + b_parab))


        # --- cria uma função para obter propriedades (com fallback) ---
        def get_gas_transport_props(Pc_bar, MR, Tc_K, default_mu=3.5e-5, default_Cp=3000.0, default_Pr=0.7, default_cstar=1500.0):
            """
            Tenta obter mu, Cp, Pr, c*.
            Se a tua instalação do rocketcea não tiver transport, cai para defaults.
            Unidades:
            mu  [Pa.s]
            Cp  [J/kg/K]
            Pr  [-]
            c*  [m/s]
            """
            mu = default_mu
            Cp = default_Cp
            Pr = default_Pr
            cstar = default_cstar

            # c* normalmente existe
            try:
                cstar = float(C.get_Cstar(Pc=Pc_bar, MR=MR))  # rocketcea: tipicamente em m/s
            except Exception:
                pass

            # Transport properties: nem todas as builds expõem isto de forma direta
            # Se tiveres métodos específicos na tua versão, mete aqui.
            # Exemplo (se existir na tua):
            # props = C.get_Chamber_Transport(Pc=Pc_bar, MR=MR)
            # mu, Cp, Pr = props["visc"], props["cp"], props["pr"]
            return mu, Cp, Pr, cstar


        # --- adiciona este bloco dentro do run_simulation(mode) (depois de já teres xs0,xs1,arc_x,x_vals etc em arrays) ---
        def run_thermal():
            """
            Calcula h_g(x), sigma(x), T_aw(x) e q''(x) ao longo do nozzle.
            Baseado em:
            q'' = h_g (T_aw - T_wg)  (6-1)
            T_aw eq (6-2)
            Bartz simplificado (6-3)
            sigma (6-4)
            """
            # --- Inputs (podes criar Entries no GUI; aqui usei defaults + tentativa de ler de entries se existirem) ---
            # Temperatura da parede no lado do gás (Twg). Se não souberes, mete um guess (ex: 600-1200K para grafite, depende do caso).
            Twg = 800.0
            r_recovery = 0.9  # fator r da eq (6-2). Podes também aproximar por ~Pr^(1/3) em turbulento, mas aqui deixo como input.

            # Se quiseres mesmo meter no GUI, cria entry_Twg e entry_r, e tenta ler:
            # Twg = float(entry_Twg.get())
            # r_recovery = float(entry_r.get())

            # --- Constantes geométricas para Bartz ---
            Dt = 2.0 * rt
            At = math.pi * rt**2

            # raio de curvatura no throat (R). No teu modelo tens r_sup = 0.4*rt (arco supersónico).
            # Se quiseres outra definição, ajusta aqui.
            R_throat = 0.4 * rt

            # Propriedades do gás
            mu, Cp, Pr, cstar = get_gas_transport_props(Pc, MR, Tc)

            # Constrói lista de estações ao longo do nozzle (ordenadas)
            x_all = np.concatenate([xs0, xs1, arc_x, x_vals])
            y_all = np.concatenate([ys0, ys1, arc_y, y_vals])

            mask = np.isfinite(x_all) & np.isfinite(y_all)
            x_all = x_all[mask]
            y_all = y_all[mask]

            idx = np.argsort(x_all)
            x_all = x_all[idx]
            y_all = y_all[idx]

            # Calcula Mach em cada estação (usa o teu find_mach, que já respeita a geometria)
            Ma_all = np.array([find_mach(float(x)) for x in x_all], dtype=float)

            # Eq (6-2): Taw
            gamma_local = gamma  # no teu código gamma vem de get_Chamber_MolWt_gamma (constante). Se quiseres gamma(x), aí já é outro nível.
            Taw = Tc * (1.0 + r_recovery * ((gamma_local - 1.0) / 2.0) * Ma_all**2) / (1.0 + ((gamma_local - 1.0) / 2.0) * Ma_all**2)

            # Eq (6-4): sigma
            sigma = (
                (0.5 * (Twg / Tc) * (1.0 + ((gamma_local - 1.0) / 2.0) * Ma_all**2) + 0.5) ** (-0.68)
                * (1.0 + ((gamma_local - 1.0) / 2.0) * Ma_all**2) ** (-0.12)
            )

            # Eq (6-3): hg (Bartz simplificado)
            # Atenção: no PDF a parte "[]" é constante ao longo do nozzle; varia com (At/Ax)^0.9 e sigma.
            # (P0g/c*) -> aqui vamos usar Pc (bar) e converter para Pa para coerência dimensional.
            # Se preferires trabalhar em bar como o teu resto, mantém mas fica "semi-empírico".
            Pc_Pa = Pc * 1e5  # bar -> Pa

            # Ax em cada estação:
            Ax = math.pi * y_all**2

            const_bracket = (
                0.026 / (Dt**0.2)
                * ((mu**0.2) * Cp / (Pr**0.6))
                * (Pc_Pa / cstar)
                * ((Dt / R_throat) ** 0.1)
            )
            hg = const_bracket * ((At / Ax) ** 0.9) * sigma  # W/m^2/K (aprox. empírico)

            # Eq (6-1): q''
            qdot = hg * (Taw - Twg)

            # Guarda outputs (para ires buscar noutro código)
            last_results["thermal"] = {
                "x_m": x_all.copy(),
                "y_m": y_all.copy(),
                "Ma": Ma_all.copy(),
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

            # Plot rápido (se quiseres)
            plt.figure(figsize=(10, 6))
            plt.plot(x_all, hg)
            plt.grid()
            plt.xlabel("x (m)")
            plt.ylabel("h_g (W/m²/K)")
            plt.title("Convective heat transfer coefficient (Bartz simplified)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(x_all, qdot)
            plt.grid()
            plt.xlabel("x (m)")
            plt.ylabel("q'' (W/m²)")
            plt.title("Convective heat flux")
            plt.tight_layout()
            plt.show()

            # opcional: CSV
            try:
                with open("thermal_profile.csv", "w") as f:
                    f.write("x_m,y_m,Ma,Taw_K,Twg_K,sigma,hg_W_m2K,qdot_W_m2\n")
                    for xi, yi, Mai, Tawi, si, hgi, qi in zip(x_all, y_all, Ma_all, Taw, sigma, hg, qdot):
                        f.write(f"{xi:.6f},{yi:.6f},{Mai:.6f},{Tawi:.3f},{Twg:.3f},{si:.6f},{hgi:.3f},{qi:.3f}\n")
                lbl_result.config(text="Thermal profile written to thermal_profile.csv", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Thermal CSV error: {e}", foreground="red")

            return last_results["thermal"]


        # --- corrige o teu theta_out e adiciona modos novos no fim do run_simulation ---
        # (coloca isto já perto do teu bloco: if mode == '2d': ...)

        # calcula theta_out corretamente (em graus) e guarda
        theta_out_deg = compute_theta_out_deg(a, b_parab, Lfinal)
        last_results["theta_out_deg"] = theta_out_deg
        last_results["a"] = float(a)
        last_results["b_parab"] = float(b_parab)
        last_results["c"] = float(c)
        last_results["Lfinal"] = float(Lfinal)

        # ============================================================
        # Robust Area–Mach relations (NO fsolve)
        # ============================================================




            


        
        



        if mode == "theta":
            # para chamares isto a partir de outro script e receberes o valor
            return theta_out_deg
        

        elif mode == "thermal":
            return run_thermal()
        
        elif mode == "friction_bl_cea":
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

            # plots (manténs)
            plot_friction_results(res, translation_x, rt)

            return res


        # (mantém os teus outros elifs como estão)

        if mode == "calibrate_k":
            # 1) compute Phi for THIS geometry (diverging only)
            Phi_ref, dbg = turning_metric_phi(x_all, r_all, translation_x, only_diverging=True)

            # 2) choose a reference turning efficiency (1–2% loss typical)
            eta_turn_ref = 0.99  # podes mudar para 0.985 se quiseres penalizar mais

            # 3) compute k
            k_cal = calibrate_k_from_reference(Phi_ref, eta_turn_ref=eta_turn_ref)

            # store
            last_results["Phi_ref"] = Phi_ref
            last_results["k_turn"] = k_cal
            last_results["eta_turn_ref"] = eta_turn_ref

            lbl_result.config(text=f"Calibrated k={k_cal:.4f} (Phi={Phi_ref:.4f} rad, eta_ref={eta_turn_ref:.3f})",
                            foreground="chartreuse4")
            return k_cal


        if mode == "score":
            # require k to exist
            k_turn = last_results.get("k_turn", 0.1)  # fallback se ainda não calibraste

            Phi, _ = turning_metric_phi(x_all, r_all, translation_x, only_diverging=True)
            eta_turn = eta_turn_from_phi(Phi, k_turn)

            # exemplo: eta_div a partir do teu theta_out_deg (tu já calculas)
            eta_div = math.cos(math.radians(abs(theta_out_deg)))

            # eta_fric do teu BL+CEA (já tens)
            fr = friction_bl_cea(x_all=x_all, r_all=r_all, rt=rt, translation_x=translation_x,
                                Tc=Tc, Pc_bar=Pc, MR=MR, exp=exp, C=C, plot_debug=False)
            eta_fric = fr["eta_v"]

            eta_total = eta_div * eta_fric * eta_turn

            last_results["eta_turn"] = eta_turn
            last_results["eta_div"] = eta_div
            last_results["eta_fric"] = eta_fric
            last_results["eta_total"] = eta_total

            lbl_result.config(text=f"score: eta_total={eta_total:.4f} (div={eta_div:.4f}, fric={eta_fric:.4f}, turn={eta_turn:.4f})",
                            foreground="chartreuse4")
            return eta_total


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
        elif mode == 'theta':
            return theta_out_deg

    except Exception as e:
        lbl_result.config(text=f"Error: {str(e)}")


# GUI Layout
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

labels = [
    ("Combustion chamber temperature (K):", "3000"),
    ("Combustion chamber pressure (bar):", "30"),
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

def plot_sensitivity_theta_in():
    # valores de k para comparar
    k_list = [0.05, 0.10, 0.20]

    # intervalo de theta_in (graus)
    theta_min = 15.0
    theta_max = 45.0
    N = 25
    theta_grid = np.linspace(theta_min, theta_max, N)

    # guarda o valor original do entry
    theta_in_original = entry_theta_in.get()

    plt.figure(figsize=(10, 6))

    for k in k_list:
        etas = []
        etas_div = []
        etas_fric = []
        etas_turn = []

        # override global do k para o score
        last_results["k_turn_override"] = float(k)

        for th in theta_grid:
            # set theta_in no GUI (em graus)
            entry_theta_in.delete(0, tk.END)
            entry_theta_in.insert(0, f"{th:.6f}")

            # corre score (não deve fazer plots)
            eta_total = run_simulation("score")

            # se houver erro e eta_total vier None
            if eta_total is None:
                etas.append(np.nan)
                etas_div.append(np.nan)
                etas_fric.append(np.nan)
                etas_turn.append(np.nan)
                continue

            etas.append(float(eta_total))
            etas_div.append(float(last_results.get("eta_div", np.nan)))
            etas_fric.append(float(last_results.get("eta_fric", np.nan)))
            etas_turn.append(float(last_results.get("eta_turn", np.nan)))

        etas = np.array(etas, dtype=float)
        plt.plot(theta_grid, etas, marker="o", label=f"k={k:.2f}")

    # repõe theta_in original e limpa override
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


    
ttk.Button(frame, text="Plot 2D", command=lambda: run_simulation('2d')).grid(row=30, column=0)
ttk.Button(frame, text="Single 3D Plot", command=lambda: run_simulation('3ds')).grid(row=30, column=1)
ttk.Button(frame, text="4 Views 3D Plot", command=lambda: run_simulation('3dm')).grid(row=31, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Update CSV", command=lambda: run_simulation('csv')).grid(row=32, columnspan=2)
ttk.Button(frame, text="Mach Curve", command=lambda: run_simulation('Mach')).grid(row=34, column=0, sticky="w", padx=45)
ttk.Button(frame, text="Temperature Profile", command=lambda: run_simulation('Temp')).grid(row=34 , column=1)
ttk.Button(frame, text="Pressure Profile", command=lambda: run_simulation('Pres')).grid(row=34 , columnspan=2)
ttk.Button(frame, text="Thermal (hg & q'')", command=lambda: run_simulation('thermal')).grid(row=35, columnspan=2)
ttk.Button(frame, text="Friction (BL+CEA mu)", command=lambda: run_simulation('friction_bl_cea')).grid(row=38, columnspan=2)
ttk.Button(frame, text="Calibrate k (turn)", command=lambda: run_simulation('calibrate_k')).grid(row=39, columnspan=2)
ttk.Button(frame, text="Sensitivity: theta_in vs k", command=plot_sensitivity_theta_in).grid(row=40, columnspan=2, pady=6)




lbl_result = ttk.Label(frame, text="")
lbl_result.grid(row=33, columnspan=2)

lbl_result = ttk.Label(frame, text="", font=("Arial", 10))
lbl_result.grid(row=33, columnspan=2)  

warning_label = tk.Label(root, text="", fg="red")
warning_label.grid(row=1, column=0, pady=10)

root.mainloop()

