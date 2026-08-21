import unittest
from dataclasses import replace

import numpy as np

from method_of_caracteristics import (
    MOCSettings,
    analyze_prescribed_nozzle,
    build_sauer_initial_line,
)
from method_of_caracteristics.numerical_methods import odd_axis_gradient
from method_of_caracteristics.physical_properties import (
    axis_compatibility_S,
    compatibility_S,
)
from nozzle_simulator import NozzleInputs, simulate
from nozzle_simulator.boundary_layer import (
    _cebeci_smith_eddy_viscosity,
    _compressible_integral_thicknesses,
    adiabatic_wall_temperature,
    compute_quick_boundary_layer,
)
from nozzle_simulator.cea import (
    _station,
    calculate_cea_properties,
    calculate_ideal_expansion_ratio,
)
from nozzle_simulator.flow import compute_flow
from nozzle_simulator.geometry import build_geometry
from nozzle_simulator.optimization import (
    OptimizationSettings,
    evaluate_contour,
    optimize_geometry,
)
from nozzle_simulator.performance import compute_performance, loss_breakdown


class GeometryTests(unittest.TestCase):
    def test_exit_area_matches_expansion_ratio(self):
        inputs = NozzleInputs()
        geometry = build_geometry(inputs)
        actual = (geometry.exit_radius_m / inputs.throat_radius_m) ** 2
        self.assertAlmostEqual(actual, inputs.expansion_ratio, places=10)

    def test_bell_fraction_changes_length(self):
        short = build_geometry(NozzleInputs(bell_fraction=0.70))
        long = build_geometry(NozzleInputs(bell_fraction=0.85))
        self.assertGreater(long.divergent_length_m, short.divergent_length_m)
        self.assertLess(long.theta_out_deg, short.theta_out_deg)

    def test_quadratic_matches_start_angle_and_derives_exit_angle(self):
        inputs = NozzleInputs(theta_in_deg=31.0)
        geometry = build_geometry(inputs)
        a, b, _ = geometry.bell_coefficients
        bell_x, _ = geometry.segments["Bell parabola"]
        local_x = bell_x - geometry.throat_x_m
        start_slope = 2.0 * a * local_x[0] + b
        exit_slope = 2.0 * a * local_x[-1] + b
        self.assertAlmostEqual(np.degrees(np.arctan(start_slope)), 31.0, places=10)
        self.assertAlmostEqual(
            np.degrees(np.arctan(exit_slope)), geometry.theta_out_deg, places=10
        )


class BoundaryLayerSignTests(unittest.TestCase):
    def test_integral_thickness_does_not_hide_reverse_flow(self):
        y = np.array([0.0, 0.5, 1.0])
        velocity = np.array([0.0, -0.5, 1.0])
        density = np.ones_like(y)
        _, momentum = _compressible_integral_thicknesses(
            y, velocity, density, edge_velocity=1.0, edge_density=1.0
        )
        self.assertLess(momentum, 0.0)

    def test_zero_cea_viscosity_is_rejected_before_log_interpolation(self):
        with self.assertRaisesRegex(ArithmeticError, "viscosity"):
            _station(
                3000.0,
                (25.0, 1.2),
                (2500.0, 0.0, 0.01, 0.7),
            )


class BoundaryLayerTests(unittest.TestCase):
    def test_cebeci_smith_eddy_viscosity_vanishes_at_boundaries(self):
        y = np.linspace(0.0, 0.01, 81)
        velocity = 800.0 * np.tanh(y / 0.001)
        density = np.full_like(y, 0.9)
        viscosity = np.full_like(y, 4.0e-5)
        eddy = _cebeci_smith_eddy_viscosity(
            y=y,
            velocity=velocity,
            density=density,
            molecular_viscosity=viscosity,
            edge_velocity=800.0,
            edge_density=0.9,
            displacement_thickness=8.0e-4,
            wall_shear=200.0,
        )
        self.assertEqual(eddy[0], 0.0)
        self.assertEqual(eddy[-1], 0.0)
        self.assertGreater(np.max(eddy[1:-1]), 0.0)


class AxisymmetricMOCTests(unittest.TestCase):
    def test_sauer_line_uses_cea_gas_properties_and_is_supersonic(self):
        inputs = NozzleInputs()
        cea = calculate_cea_properties(inputs)
        gas_constant = 8.314462618 / (cea.throat.molecular_weight_g_mol / 1000.0)
        line = build_sauer_initial_line(
            stagnation_pressure_pa=inputs.chamber_pressure_bar * 1.0e5,
            stagnation_temperature_k=cea.chamber.temperature_k,
            gamma=cea.throat.gamma,
            gas_constant_j_kg_k=gas_constant,
            throat_radius_m=inputs.throat_radius_m,
            throat_x_m=build_geometry(inputs).throat_x_m,
            radial_stations=21,
        )
        self.assertEqual(line.radius_m.size, 21)
        self.assertTrue(np.all(line.mach > 1.0))
        self.assertAlmostEqual(line.theta_rad[0], 0.0, places=12)
        self.assertAlmostEqual(line.theta_rad[-1], 0.0, places=10)
        self.assertAlmostEqual(
            line.subsonic_curvature_radius_m,
            1.5 * inputs.throat_radius_m,
            places=12,
        )
        self.assertGreater(line.x_m[0], line.x_m[-1])

    def test_axis_source_limit_uses_odd_theta_expansion(self):
        M = 2.4
        c_1 = 3.2
        radius = np.array([0.0, 1.0e-5, 2.0e-5])
        theta = c_1 * radius - 0.8 * radius**3
        estimated_gradient = odd_axis_gradient(radius, theta)
        self.assertAlmostEqual(estimated_gradient, c_1, places=9)
        expected = axis_compatibility_S(M, c_1)
        near_axis = compatibility_S(M, theta[1], radius[1], "minus")
        self.assertAlmostEqual(near_axis, expected, places=4)

    def test_prescribed_wall_moc_returns_finite_axisymmetric_field(self):
        inputs = NozzleInputs()
        cea = calculate_cea_properties(inputs)
        geometry = build_geometry(inputs)
        result = analyze_prescribed_nozzle(
            inputs,
            geometry,
            cea,
            settings=MOCSettings(
                axial_stations=60,
                radial_stations=15,
                initialization="quasi_1d",
                start_mach=1.16,
            ),
        )
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.mach).all())
        self.assertTrue(np.isfinite(result.pressure_pa).all())
        np.testing.assert_allclose(result.radius_m[:, 0], 0.0)
        np.testing.assert_allclose(result.theta_rad[:, 0], 0.0)
        self.assertGreater(result.inviscid_thrust_coefficient, 0.0)
        self.assertGreater(result.mass_flow_residual, 0.0)

    def test_sauer_initialized_moc_reaches_the_exit(self):
        inputs = NozzleInputs()
        result = analyze_prescribed_nozzle(
            inputs,
            build_geometry(inputs),
            calculate_cea_properties(inputs),
            settings=MOCSettings(
                axial_stations=120,
                radial_stations=21,
                initialization="sauer",
                corrector_iterations=30,
            ),
        )
        self.assertTrue(result.converged)
        self.assertIn("Sauer", result.initialization)
        self.assertEqual(result.initial_line_x_m.size, 21)
        self.assertTrue(np.isfinite(result.mach).all())
        self.assertGreater(result.total_time_s, 0.0)

    def test_kliegel_levine_curved_line_matches_cea_choked_flow(self):
        from method_of_caracteristics import (
            build_kliegel_levine_initial_line,
            curved_line_mass_flow_kg_s,
        )

        inputs = NozzleInputs()
        cea = calculate_cea_properties(inputs)
        geometry = build_geometry(inputs)
        gas_constant = 8.314462618 / (
            cea.throat.molecular_weight_g_mol / 1000.0
        )
        line = build_kliegel_levine_initial_line(
            stagnation_pressure_pa=inputs.chamber_pressure_bar * 1.0e5,
            stagnation_temperature_k=cea.chamber.temperature_k,
            gamma=cea.throat.gamma,
            gas_constant_j_kg_k=gas_constant,
            throat_radius_m=inputs.throat_radius_m,
            throat_x_m=geometry.throat_x_m,
            radial_stations=31,
        )
        cea_mass_flow = (
            inputs.chamber_pressure_bar
            * 1.0e5
            * np.pi
            * inputs.throat_radius_m**2
            / cea.cstar_m_s
        )
        self.assertTrue(np.isfinite(line.mach).all())
        self.assertTrue(np.all(line.mach > 1.0))
        self.assertGreater(line.discharge_coefficient, 0.98)
        self.assertLess(line.discharge_coefficient, 1.0)
        self.assertLess(
            abs(curved_line_mass_flow_kg_s(line) / cea_mass_flow - 1.0), 0.01
        )

    def test_kliegel_levine_transition_reaches_fixed_x_marcher(self):
        inputs = NozzleInputs()
        result = analyze_prescribed_nozzle(
            inputs,
            build_geometry(inputs),
            calculate_cea_properties(inputs),
            settings=MOCSettings(
                axial_stations=120,
                radial_stations=21,
                initialization="kliegel_levine",
            ),
        )
        self.assertTrue(result.converged)
        self.assertIn("Kliegel-Levine", result.initialization)
        self.assertGreater(len(result.transition_line_x_m), 3)
        self.assertEqual(
            len(result.transition_line_x_m), len(result.transition_line_radius_m)
        )
        self.assertLess(result.mass_flow_residual, 0.05)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = simulate(NozzleInputs())

    def test_cea_values_are_physical(self):
        self.assertGreater(self.result.cea.chamber.temperature_k, 1000.0)
        self.assertGreater(self.result.cea.cstar_m_s, 1000.0)
        self.assertGreater(self.result.cea.exit_mach, 1.0)

    def test_ambient_pressure_is_converted_from_atm(self):
        self.assertAlmostEqual(NozzleInputs().ambient_pressure_bar, 1.01325, places=8)

    def test_ideal_expansion_ratio_matches_ambient_pressure(self):
        inputs = NozzleInputs()
        ideal_eps = self.result.cea.ideal_expansion_ratio
        matched = calculate_cea_properties(replace(inputs, expansion_ratio=ideal_eps))
        self.assertAlmostEqual(matched.exit_pressure_bar, inputs.ambient_pressure_bar, places=6)

    def test_effective_cf_decomposition(self):
        result = self.result
        inputs, cea, performance = result.inputs, result.cea, result.performance
        momentum_cf = cea.ideal_momentum_thrust_coefficient
        pressure_cf = inputs.expansion_ratio * (
            cea.exit_pressure_bar - inputs.ambient_pressure_bar
        ) / inputs.chamber_pressure_bar
        expected = (
            performance.momentum_efficiency * momentum_cf
            + pressure_cf
            - performance.friction_thrust_coefficient
        )
        self.assertAlmostEqual(performance.effective_thrust_coefficient, expected, places=12)

    def test_loss_breakdown_is_non_negative_and_closes_momentum_losses(self):
        performance = self.result.performance
        losses = loss_breakdown(performance)
        self.assertTrue(all(value >= 0.0 for value in losses.values()))
        expected_momentum_loss = (
            1.0 - performance.momentum_efficiency
        ) * performance.momentum_thrust_coefficient
        self.assertAlmostEqual(
            losses["Exit divergence"] + losses["BL displacement"],
            expected_momentum_loss,
            places=12,
        )

    def test_profiles_are_aligned_and_finite(self):
        result = self.result
        size = len(result.geometry.x_m)
        arrays = (
            result.geometry.radius_m,
            result.flow.mach,
            result.flow.temperature_k,
            result.flow.pressure_bar,
            result.thermal.heat_flux_w_m2,
            result.boundary_layer.displacement_thickness_m,
            result.boundary_layer.momentum_thickness_m,
            result.boundary_layer.shape_factor,
            result.boundary_layer.skin_friction_coefficient,
        )
        self.assertTrue(all(len(array) == size for array in arrays))
        self.assertTrue(all(np.isfinite(array).all() for array in arrays))

    def test_blimp_lite_boundary_layer_closure(self):
        boundary_layer = self.result.boundary_layer
        self.assertTrue(np.all(boundary_layer.momentum_thickness_m >= 0.0))
        np.testing.assert_allclose(
            boundary_layer.displacement_thickness_m,
            boundary_layer.shape_factor * boundary_layer.momentum_thickness_m,
            rtol=1e-12,
            atol=1e-14,
        )
        self.assertTrue(np.all(boundary_layer.skin_friction_coefficient > 0.0))
        expected_cf = 2.0 * boundary_layer.wall_shear_stress_pa / (
            self.result.flow.pressure_bar
            * 1e5
            / (
                (
                    8.314462618
                    / (self.result.cea.exit.molecular_weight_g_mol / 1000.0)
                )
                * self.result.flow.temperature_k
            )
            * (
                self.result.flow.mach
                * np.sqrt(
                    self.result.flow.gamma
                    * (
                        8.314462618
                        / (self.result.cea.exit.molecular_weight_g_mol / 1000.0)
                    )
                    * self.result.flow.temperature_k
                )
            )
            ** 2
        )
        # Molecular weight is interpolated through the nozzle, so compare at exit.
        self.assertAlmostEqual(
            boundary_layer.skin_friction_coefficient[-1], expected_cf[-1], places=10
        )

    def test_wall_is_adiabatic_recovery_temperature(self):
        expected = adiabatic_wall_temperature(
            self.result.geometry, self.result.flow, self.result.cea
        )
        np.testing.assert_allclose(self.result.thermal.wall_temperature_k, expected)
        np.testing.assert_allclose(self.result.boundary_layer.wall_temperature_k, expected)
        np.testing.assert_allclose(self.result.thermal.heat_flux_w_m2, 0.0)

    def test_short_optimizer_run_returns_valid_design(self):
        base = NozzleInputs(theta_sub_deg=57.0)
        optimized = optimize_geometry(
            base, num_generations=1, population_size=6
        )
        self.assertGreaterEqual(optimized.bell_fraction, 0.60)
        self.assertLessEqual(optimized.bell_fraction, 1.00)
        self.assertAlmostEqual(
            optimized.expansion_ratio,
            calculate_ideal_expansion_ratio(base),
            places=10,
        )
        self.assertGreaterEqual(optimized.theta_in_deg, 20.0)
        self.assertLessEqual(optimized.theta_in_deg, 35.0)
        self.assertEqual(optimized.theta_sub_deg, base.theta_sub_deg)
        self.assertGreaterEqual(optimized.theta_out_deg, 0.0)
        self.assertLess(optimized.theta_out_deg, optimized.theta_in_deg)
        self.assertGreater(optimized.fitness, 0.0)

    def test_fast_fitness_matches_full_simulation_merit(self):
        inputs = NozzleInputs()
        solution = (0.82, 29.0)
        design = NozzleInputs(
            bell_fraction=solution[0],
            theta_in_deg=solution[1],
        )
        result = simulate(design)
        expected = result.performance.effective_thrust_coefficient
        self.assertAlmostEqual(evaluate_contour(inputs, solution), expected, places=12)

    def test_quick_optimizer_evaluator_uses_weak_integral_closure(self):
        inputs = NozzleInputs()
        solution = (0.82, 29.0)
        design = replace(
            inputs,
            bell_fraction=solution[0],
            theta_in_deg=solution[1],
        )
        cea = calculate_cea_properties(design)
        geometry = build_geometry(design)
        flow = compute_flow(design, geometry, cea)
        boundary_layer = compute_quick_boundary_layer(design, geometry, flow, cea)
        expected = compute_performance(
            design, geometry, cea, boundary_layer
        ).effective_thrust_coefficient
        self.assertAlmostEqual(
            evaluate_contour(inputs, solution, "quick"), expected, places=12
        )
        self.assertTrue(np.isfinite(boundary_layer.skin_friction_coefficient).all())

    def test_process_evaluation_mode_returns_valid_design(self):
        settings = OptimizationSettings(
            num_generations=1,
            population_size=6,
            num_parents_mating=3,
            keep_elitism=1,
            evaluation_mode="processes",
            parallel_workers=2,
        )
        optimized = optimize_geometry(NozzleInputs(), settings=settings)
        self.assertGreater(optimized.fitness, 0.0)

    def test_quick_optimization_mode_returns_screening_result(self):
        settings = OptimizationSettings(
            num_generations=1,
            population_size=6,
            num_parents_mating=3,
            keep_elitism=1,
            evaluation_mode="serial",
            boundary_layer_model="quick",
        )
        optimized = optimize_geometry(NozzleInputs(), settings=settings)
        self.assertEqual(optimized.boundary_layer_model, "quick")
        self.assertGreater(optimized.fitness, 0.0)


if __name__ == "__main__":
    unittest.main()
