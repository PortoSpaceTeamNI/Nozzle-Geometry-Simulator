"""Bell-nozzle geometry extracted from the original NozzleGeometry model."""

import math

import numpy as np

from .models import GeometryResult, NozzleInputs


def build_geometry(inputs: NozzleInputs) -> GeometryResult:
    inputs.validate()
    rt = inputs.throat_radius_m
    chamber_radius = inputs.chamber_diameter_m / 2.0
    theta_sub = math.radians(inputs.theta_sub_deg)
    theta_in = math.radians(inputs.theta_in_deg)
    alpha = math.radians(inputs.reference_half_angle_deg)

    # Convergent: straight line tangent to a 1.5 Rt subsonic throat arc.
    xr = -1.5 * rt * math.sin(theta_sub)
    yr = 2.5 * rt - 1.5 * rt * math.cos(theta_sub)
    if chamber_radius < yr:
        raise ValueError("Chamber diameter is too small for the convergent angle.")
    line_slope = -math.tan(theta_sub)
    line_intercept = yr - line_slope * xr
    xf = (chamber_radius - line_intercept) / line_slope
    throat_x = -xf

    line_x0 = np.linspace(xf, xr, 120)
    line_r = line_slope * line_x0 + line_intercept
    sub_x0 = np.linspace(xr, 0.0, 120)
    sub_r = 2.5 * rt - np.sqrt(np.maximum(2.25 * rt**2 - sub_x0**2, 0.0))

    # Divergent: 0.4 Rt throat arc followed by a quadratic bell.
    exit_radius = math.sqrt(inputs.expansion_ratio) * rt
    cone_length = (exit_radius - rt) / math.tan(alpha)
    bell_length = inputs.bell_fraction * cone_length
    sup_angles = np.linspace(math.pi / 2.0, math.pi / 2.0 - theta_in, 100)
    sup_x0 = 0.4 * rt * np.cos(sup_angles)
    sup_r = 1.4 * rt - 0.4 * rt * np.sin(sup_angles)
    px, py = float(sup_x0[-1]), float(sup_r[-1])
    exit_x0 = px + bell_length

    matrix = np.array([
        [px**2, px, 1.0],
        [2.0 * px, 1.0, 0.0],
        [exit_x0**2, exit_x0, 1.0],
    ])
    rhs = np.array([py, math.tan(theta_in), exit_radius])
    a, b, c = np.linalg.solve(matrix, rhs)
    if a > 0.0:
        raise ValueError("Invalid bell contour: the parabola has inverted concavity.")

    parab_x0 = np.linspace(px, exit_x0, 300)
    parab_r = a * parab_x0**2 + b * parab_x0 + c
    wall_slope = 2.0 * a * parab_x0 + b
    if np.any(parab_r > exit_radius + 1e-4):
        raise ValueError("Invalid bell contour: the parabola exceeds the exit radius.")
    if np.any(np.diff(parab_r) < -1e-9):
        raise ValueError("Invalid bell contour: wall radius decreases in the divergent.")
    if np.any(wall_slope < -1e-9):
        raise ValueError("Invalid bell contour: the local wall angle becomes negative.")

    theta_out_calculated = math.degrees(math.atan(wall_slope[-1]))
    if theta_out_calculated < 0.0:
        raise ValueError("Invalid bell contour: the exit wall angle is negative.")

    translate = throat_x
    segments = {
        "Convergent line": (line_x0 + translate, line_r),
        "Subsonic arc": (sub_x0 + translate, sub_r),
        "Supersonic arc": (sup_x0 + translate, sup_r),
        "Bell parabola": (parab_x0 + translate, parab_r),
    }
    x_all = np.concatenate([segment[0] for segment in segments.values()])
    r_all = np.concatenate([segment[1] for segment in segments.values()])
    order = np.argsort(x_all)
    x_all, r_all = x_all[order], r_all[order]
    _, unique_indices = np.unique(np.round(x_all, 12), return_index=True)
    x_all = x_all[unique_indices]
    r_all = r_all[unique_indices]

    exit_x = exit_x0 + translate
    return GeometryResult(
        x_m=x_all,
        radius_m=r_all,
        segments=segments,
        throat_x_m=throat_x,
        exit_x_m=exit_x,
        exit_radius_m=exit_radius,
        divergent_length_m=exit_x0,
        total_length_m=exit_x,
        cone_length_m=cone_length,
        theta_out_deg=theta_out_calculated,
        contraction_ratio=(chamber_radius / rt) ** 2,
        bell_coefficients=(float(a), float(b), float(c)),
    )
