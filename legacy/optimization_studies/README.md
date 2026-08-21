# Archived optimization studies

These files preserve the standalone genetic-algorithm experiments that predate the
shared simulator package.

| Technical filename | Previous filename | Purpose |
|---|---|---|
| `reduced_order_nozzle_performance.py` | `apropulsive_performance_model.py` | Independent reduced-order performance model |
| `specific_impulse_objective.py` | `fitness_function_specific_impulse.py` | Specific-impulse merit and constraints |
| `specific_impulse_ga_study.py` | `ga_optimizer_isp.py` | Standalone Isp optimization study |
| `total_efficiency_objective.py` | `fitness_function_final.py` | Total-efficiency merit and constraints |
| `total_efficiency_ga_study.py` | `ga_optimizer.py` | Standalone total-efficiency study |

They are kept for result traceability and are not the maintained application entry
point. The active optimizer is:

```text
nozzle_simulator/optimization/contour_genetic_algorithm.py
```

It runs directly inside the main simulator window and evaluates candidates through
the same simulation modules used by the displayed plots.

