"""Minimal non-GUI example."""

from nozzle_simulator import NozzleInputs, simulate

result = simulate(
    NozzleInputs(
        chamber_pressure_bar=30.0,
        mixture_ratio=6.5,
        ambient_pressure_atm=1.0,
        expansion_ratio=5.6,
        throat_radius_m=0.01728,
        chamber_diameter_m=0.120,
        bell_fraction=0.80,
        theta_in_deg=30.0,
        theta_sub_deg=50.0,
    )
)

print(f"CEA chamber temperature: {result.cea.chamber.temperature_k:.1f} K")
print(f"Exit angle: {result.geometry.theta_out_deg:.3f} deg")
print(f"Boundary-layer velocity efficiency: {result.boundary_layer.velocity_efficiency:.5f}")
print(f"Effective ambient Cf: {result.performance.effective_thrust_coefficient:.5f}")
