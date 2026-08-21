"""Regression tests for the pre-sized bell-nozzle optimizer."""

import unittest

import numpy as np

from genetic_algorithm.apropulsive_performance_model import (
    compute_exit_radius,
    compute_parabola_coefficients,
)
from genetic_algorithm.fitness_function_specific_impulse import (
    EPS_FIXED,
    R_T_FIXED,
    evaluate_solution,
)


class BellContourTests(unittest.TestCase):
    def test_parabola_reaches_fixed_exit_radius(self):
        a, b, c, _, length = compute_parabola_coefficients(
            R_T_FIXED, EPS_FIXED, 30.0, 0.8)
        radius = a * length**2 + b * length + c
        self.assertAlmostEqual(radius, compute_exit_radius(R_T_FIXED, EPS_FIXED), places=12)

    def test_reported_exit_angle_is_derived_from_contour(self):
        _, result = evaluate_solution([0.8, 30.0, 50.0])
        slope_exit = 2.0 * result["a"] * result["L [m]"] + result["b"]
        expected = np.degrees(np.arctan(slope_exit))
        self.assertAlmostEqual(result["theta_out [°]"], expected, places=12)

    def test_longer_bell_increases_integrated_wall_friction(self):
        _, short = evaluate_solution([0.7, 30.0, 50.0])
        _, long = evaluate_solution([0.9, 30.0, 50.0])
        self.assertGreater(long["friction_force [N]"], short["friction_force [N]"])
        self.assertGreater(long["eta_div"], short["eta_div"])

    def test_fitness_is_finite_for_nominal_design(self):
        fitness, result = evaluate_solution([0.8, 30.0, 50.0])
        self.assertTrue(np.isfinite(fitness))
        self.assertGreater(fitness, 0.0)
        self.assertNotIn("eta_turn", result)


if __name__ == "__main__":
    unittest.main()
