import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math

# Create main window
root = tk.Tk()
root.title("Nozzle Profile Generator")

# Main simulation function
def run_simulation(mode):
    try:
        # User Inputs
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
            total_points = 100  # ou outro valor que preferires

            # Distributing the Points Equally
            lengths = [len(arr) for arr in x_combined]
            total_len = sum(lengths)

            x_raw = []
            y_raw = []

            for x_seg, y_seg, seg_len in zip(x_combined, y_combined, lengths):
                # Creating Step and Deffining a Minimum Step (=2)
                n_points = max(2, int((seg_len / total_len) * total_points))
                # Creating index (int) Equally Spaced
                indices = np.linspace(0, seg_len - 1, n_points).astype(int)

                x_raw.extend([x_seg[i] for i in indices])
                y_raw.extend([y_seg[i] for i in indices])

            # Minimum Distance in mm
            min_dist_mm = 1
    	    
            # Checking if There's Points to Add
            if len(x_raw) > 0:
                # If There's Points, the First Points is Always Accepted and Added to the Filtered List
                x_filtered = [x_raw[0]]
                y_filtered = [y_raw[0]]
                last_x = x_raw[0]
                last_y = y_raw[0]

                # To Filter the Rest of the Points, a the Rest of the Points are Filtered (Expect the First Points, Already Accepted)
                for xi, yi in zip(x_raw[1:], y_raw[1:]):
                    # Getting the Distance Between the Actual Point and the Last Point
                    dist = math.hypot((xi - last_x) * 1000, (yi - last_y) * 1000) 
                    if dist >= min_dist_mm:
                        # If the Distance is Bigger than the Minimum Distance the Point is Added and the Last Point is Updated
                        x_filtered.append(xi)
                        y_filtered.append(yi)
                        last_x = xi
                        last_y = yi

                x_final = x_filtered
                y_final = y_filtered

            # Ensuring the Last Point is Correct (Lfinal, re)
            if x_final[-1] != Lfinal or y_final[-1] != re:
                x_final[-1] = round(Lfinal + translation_x, 4)
                y_final[-1] = round(re, 4)

            # Converting m to mm
            x_final_mm = [x * 1000 for x in x_final]
            y_final_mm = [y * 1000 for y in y_final]

            # Zipping and Sorting the Points
            combined = list(zip(x_final_mm, y_final_mm))
            combined.sort()

            x_final_sorted, y_final_sorted = zip(*combined)

            try:
                with open("nozzle_geometry.csv", "w") as file:
                    file.write("x,y,z\n")
                    for x, y in zip(x_final_sorted, y_final_sorted):
                        file.write(f"{round(x,2)},{round(y,2)},0\n")
                lbl_result.config(text="CSV Updated Successfully!", foreground="chartreuse4")
            except Exception as e:
                lbl_result.config(text=f"Erro: {str(e)}", foreground="red")




        if mode == '2d':
            plot2d()
        elif mode == '3dm':
            plot3d_multi()
        elif mode == '3ds':
            plot3d_single()
        elif mode == 'csv':
            update_csv()

    except Exception as e:
        lbl_result.config(text=f"Error: {str(e)}")


# GUI Layout
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

labels = [
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

entry_rt, entry_exp, entry_halfangle, entry_theta_in, entry_theta_sub, entry_Rchamber, entry_bell_contour = entries

import time

# Debounce
last_update_time = 0
debounce_delay = 500 
scheduled_update_id = None

def on_input_change(event=None):
    global last_update_time, scheduled_update_id
    now = time.time() * 1000  # Em ms
    if scheduled_update_id:
        root.after_cancel(scheduled_update_id)
    scheduled_update_id = root.after(debounce_delay, lambda: run_simulation('2d'))

for entry in entries:
    entry.bind('<KeyRelease>', on_input_change)
    
ttk.Button(frame, text="Plot 2D", command=lambda: run_simulation('2d')).grid(row=30, column=0)
ttk.Button(frame, text="Single 3D Plot", command=lambda: run_simulation('3ds')).grid(row=30, column=1)
ttk.Button(frame, text="4 Views 3D Plot", command=lambda: run_simulation('3dm')).grid(row=31, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Update CSV", command=lambda: run_simulation('csv')).grid(row=32, columnspan=2)

lbl_result = ttk.Label(frame, text="")
lbl_result.grid(row=33, columnspan=2)

lbl_result = ttk.Label(frame, text="", font=("Arial", 10))
lbl_result.grid(row=33, columnspan=2)  

warning_label = tk.Label(root, text="", fg="red")
warning_label.grid(row=1, column=0, pady=10)

root.mainloop()


### MELHORAS 
# Equacoes Parabola e Arco para Funcao