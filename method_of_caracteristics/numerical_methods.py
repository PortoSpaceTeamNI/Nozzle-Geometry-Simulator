"""Numerical primitives for pressure-based axisymmetric characteristics."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .physical_properties import mach_angle, prandtl_meyer


def mach_from_prandtl_meyer(nu_target: float, gamma: float) -> float:
    return float(
        brentq(
            lambda M: prandtl_meyer(M, gamma) - nu_target,
            1.0 + 1.0e-10,
            50.0,
        )
    )


def characteristic_slope(M: float, theta: float, family: str) -> float:
    try:
        mu = mach_angle(M)
    except ValueError as exc:
        raise ValueError(
            f"Cannot construct a {family} characteristic for M={M!r}, theta={theta!r}."
        ) from exc
    if family == "plus":
        return math.tan(theta + mu)
    if family == "minus":
        return math.tan(theta - mu)
    raise ValueError("Characteristic family must be 'plus' or 'minus'.")


def intersect_characteristics(
    A: tuple[float, float, float, float, float],
    B: tuple[float, float, float, float, float],
) -> tuple[float, float]:
    """Intersect C- from A with C+ from B using endpoint slopes."""
    x_A, r_A, _p_A, M_A, theta_A = A
    x_B, r_B, _p_B, M_B, theta_B = B
    slope_A_minus = characteristic_slope(M_A, theta_A, "minus")
    slope_B_plus = characteristic_slope(M_B, theta_B, "plus")
    denominator = slope_A_minus - slope_B_plus
    if abs(denominator) < 1.0e-12:
        raise ArithmeticError("The C- and C+ characteristics are approximately parallel.")
    x_P = (
        r_B - r_A + slope_A_minus * x_A - slope_B_plus * x_B
    ) / denominator
    r_P = r_A + slope_A_minus * (x_P - x_A)
    return float(x_P), float(r_P)


def odd_axis_gradient(radius: np.ndarray, theta: np.ndarray) -> float:
    """Estimate dtheta/dr at the axis using theta = c1 r + c3 r^3."""
    radius = np.asarray(radius, dtype=float)
    theta = np.asarray(theta, dtype=float)
    positive = np.flatnonzero(radius > 0.0)
    if positive.size == 0:
        raise ValueError("At least one positive-radius point is required.")
    first = int(positive[0])
    if positive.size == 1:
        return float(theta[first] / radius[first])
    second = int(positive[1])
    r_1, r_2 = float(radius[first]), float(radius[second])
    theta_1, theta_2 = float(theta[first]), float(theta[second])
    denominator = r_2 * r_2 - r_1 * r_1
    if abs(denominator) < 1.0e-20:
        return theta_1 / r_1
    return float(
        (
            r_2 * r_2 * theta_1 / r_1
            - r_1 * r_1 * theta_2 / r_2
        )
        / denominator
    )
