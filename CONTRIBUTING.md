# Contributing

Thank you for improving the Rocket Nozzle Simulator.

## Development setup

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

## Design rules

- `nozzle_simulator.simulation.simulate()` is the single maintained entry point.
- Do not duplicate geometry or CEA equations inside the GUI or optimizer.
- Keep operating conditions, pre-sizing values and contour design variables distinct.
- Use explicit units in names, labels and exported column headers.
- Add a regression test for changes to geometry, property conversion or model output.
- Document new closures and their validity range in `docs/MODEL.md`.

## Pull requests

Keep changes focused, describe the physical assumption being modified and include the
input case used for validation. Numerical model changes should show before/after
results and cite an appropriate source where possible.

