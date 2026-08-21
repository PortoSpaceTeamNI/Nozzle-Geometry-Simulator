"""Single simulation entry point shared by the GUI and future optimizers."""

from datetime import datetime, timezone

from .boundary_layer import compute_boundary_layer
from .cea import calculate_cea_properties
from .flow import compute_flow
from .geometry import build_geometry
from .models import NozzleInputs, SimulationResult
from .performance import compute_performance
from .thermal import compute_thermal


def simulate(inputs: NozzleInputs) -> SimulationResult:
    inputs.validate()
    cea = calculate_cea_properties(inputs)
    geometry = build_geometry(inputs)
    flow = compute_flow(inputs, geometry, cea)
    thermal = compute_thermal(inputs, geometry, flow, cea)
    boundary_layer = compute_boundary_layer(inputs, geometry, flow, cea)
    performance = compute_performance(inputs, geometry, cea, boundary_layer)
    return SimulationResult(
        inputs=inputs,
        cea=cea,
        geometry=geometry,
        flow=flow,
        thermal=thermal,
        boundary_layer=boundary_layer,
        performance=performance,
        metadata={"created_utc": datetime.now(timezone.utc).isoformat()},
    )
